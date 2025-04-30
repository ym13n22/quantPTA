from causis_api.const import get_version
from causis_api.const import login
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import talib
import os

# 获取 AM 系统的账号和密码
login.username = input("Enter Username: ")  # 通过 input() 获取用户名
login.password = input("Enter Password: ")  # 通过 input() 获取密码


import pandas as pd
import numpy as np
from causis_api.data import *


symbol="R.CN.CZC.TA.0004"
start_data="2006-12-18"
end_data="2025-04-06"

windows_1 = 5

windows_2 = 40

windows_3 = 20

windows_4 = 60

short_range=[3,5,10,15,20,25,30,35,40,45,50,60,70,80,90,100]

long_range=[30,35,40,50,60,70,80,100,130,150,180,200,300,400,500,600,700,800,900,1000]

charge = 0.0002

loss_limit = -0.008


productCode = [symbol]

begin = "2006-12-18"

end = "2025-04-06"

allocation = 10000000

slippage = 1

multi=1
mintick=0.01

symbols=["R.CN.CFE.IC.0004",
"R.CN.CFE.IF.0004",
"R.CN.CFE.IH.0004",
"R.CN.CZC.CF.0004",
"R.CN.CZC.FG.0004",
"R.CN.CZC.MA.0004",
"R.CN.CZC.OI.0004",
"R.CN.CZC.RM.0004",
"R.CN.CZC.SA.0004",
"R.CN.CZC.SF.0004",
"R.CN.CZC.SM.0004",
"R.CN.CZC.SR.0004",
"R.CN.CZC.TA.0004",
"R.CN.CZC.WH.0004",
"R.CN.CZC.ZC.0004",
"R.CN.DCE.a.0004",
"R.CN.DCE.c.0004",
"R.CN.DCE.cs.0004",
"R.CN.DCE.eg.0004",
"R.CN.DCE.i.0004",
"R.CN.DCE.j.0004",
"R.CN.DCE.jd.0004",
"R.CN.DCE.jm.0004",
"R.CN.DCE.l.0004",
"R.CN.DCE.m.0004",
"R.CN.DCE.p.0004",
"R.CN.DCE.pp.0004",
"R.CN.DCE.v.0004",
"R.CN.DCE.y.0004",
"R.CN.SHF.ag.0004",
"R.CN.SHF.al.0004",
"R.CN.SHF.au.0004",
"R.CN.SHF.bu.0004",
"R.CN.SHF.cu.0004",
"R.CN.SHF.hc.0004",
"R.CN.SHF.ni.0004",
"R.CN.SHF.pb.0004",
"R.CN.SHF.rb.0004",
"R.CN.SHF.ru.0004",
"R.CN.SHF.sn.0004",
"R.CN.SHF.zn.0004"]




def load_data(symbol,start_data,end_data):
    # 获取指定标的（symbol）从开始日期（start_data）到结束日期（end_data）的历史数据
    data=get_price(symbol,  start_data,end_data  )
    #print(data.head())

    df = pd.DataFrame(data)

    # 保存为 CSV 文件
    #df.to_csv('pta_contract_data.csv', index=False, encoding='utf-8')

    #print("successfully saved data")
    return data


def load_single_csv(folder_path, symbol):
    """
    根据给定的文件名（商品标的）从指定文件夹中加载CSV文件。

    :param folder_path: 存储CSV文件的文件夹路径
    :param symbol: 商品标的的名称（即文件名，不包含扩展名）
    :return: 对应标的的DataFrame，如果找不到该文件返回None
    """
    file_path = os.path.join(folder_path, f"{symbol}.csv")

    if os.path.exists(file_path):
        # 读取CSV文件并返回DataFrame
        return pd.read_csv(file_path)
    else:
        print(f"文件 {symbol}.csv 不存在。")
        return None


def generate_signals(df, windows_1, windows_2):
    """
    生成基于均线交叉的交易信号

    参数：
    df : pd.DataFrame
        输入的历史数据 DataFrame，要求包含 "CLOSE" 列，表示每个时间点的收盘价。

    windows_1 : int
        短期均线的窗口大小，例如：5日均线。

    windows_2 : int
        长期均线的窗口大小，例如：20日均线。

    返回：
    pd.DataFrame
        返回包含信号列（"signal"）、短期均线列（"ma1"）和长期均线列（"ma2"）的 DataFrame。
    """

    df["ma1"] = df["CLOSE"].rolling(windows_1).mean()
    df["ma2"] = df["CLOSE"].rolling(windows_2).mean()
    df["signal"] = 0
    df.loc[df["ma1"] > df["ma2"], "signal"] = 1
    df.loc[df["ma1"] < df["ma2"], "signal"] = -1
    return df



def run_backtest(df, charge, loss_limit, slippage,allocation,multi,mintick):
    """
    执行回测逻辑：根据交易信号进行买卖，考虑手续费、滑点、止损等因素，计算每日收益和账户余额。

    参数：
    df : pd.DataFrame
        包含市场数据和交易信号的DataFrame，要求包含 'OPEN', 'CLOSE', 'signal' 等列。

    charge : float
        手续费率，按每次交易收取的比例费用。

    loss_limit : float
        止损阈值，若某日策略收益小于该值，则清仓止损。

    slippage : float
        每次交易的滑点比例（模拟成交价格偏离理想价格的损耗）。

    allocation : float
        初始账户资金分配，即策略初始的本金。

    multi : int
        合约乘数，用于计算实际盈亏（如每点波动对应的盈亏金额）。

    mintick : float
        最小变动价位，结合multi计算滑点损失。

    返回：
    df : pd.DataFrame
        包含回测过程中计算的各类收益、仓位、信号、账户余额等信息。

    trade_orders : pd.DataFrame
        回测过程中所有交易的时间点及信息（时间、价格、交易类型等）。
    """
    # 初始仓位，基于信号决定是否持仓
    df["position"] = df["signal"].shift(1).fillna(0)  # 使用昨日的信号作为今天的仓位
    df["return"] = df["OPEN"].pct_change().fillna(0)  # 收益率
    df["strategy_return"] = df["return"] * df["position"]  # 策略的每日收益

    # 交易滑点成本（直接使用滑点比例）
    df["trade"] = df["position"].diff().abs()  # 计算每日的交易数量
    df["slip_cost"] = df["trade"] * slippage * multi * mintick  # 计算滑点成本
    df["fee"] = df["trade"] * charge  # 手续费
    df["strategy_return"] -= (df["fee"] + df["slip_cost"])  # 扣除手续费与滑点

    # 记录买入/卖出的类型
    df["order_type"] = None  # 初始化为None
    df.loc[(df["position"] == 1) & (df["position"].shift(1) != 1), "order_type"] = "buy"  # 买入
    df.loc[(df["position"] == 0) & (df["position"].shift(1) == 1), "order_type"] = "sell"  # 卖出
    df.loc[(df["position"] == 0) & (df["position"].shift(1) == -1), "order_type"] = "buy_to_cover"  # 平空操作

    # **执行止损操作，并记录交易**
    # 止损：如果策略的收益小于止损阈值，清仓（即仓位置为0）
    df.loc[df["strategy_return"] < loss_limit, "position"] = 0

    # **止损后记录卖出**
    # 如果止损操作发生（仓位被清除），记录为 "sell" 操作
    df.loc[(df["position"] == 0) & (df["position"].shift(1) == 1), "order_type"] = "sell"

    df["trade"] = df["position"].diff().abs()

    # 止损后重新计算策略的收益
    df["strategy_return"] = df["return"] * df["position"]

    # 累计收益计算
    df["cum_return"] = (1 + df["strategy_return"]).cumprod()  # 策略累计收益
    df["interPNL"] = allocation * df["strategy_return"]  # 每日盈亏
    df["balance"] = allocation + df["interPNL"].cumsum()  # 账户余额

    # 保存每日盈亏和账户余额数据到 CSV 文件
    pnl_data = df[["CLOCK", "interPNL", "balance"]]
    pnl_data.to_csv("pnl.csv", index=False, encoding="utf-8")

    # 生成交易订单列表
    trade_orders = df.loc[df["order_type"].notna(), ["CLOCK", "CLOSE", "order_type", "position", "trade"]]

    # 返回更新后的DataFrame以及生成的交易订单列表
    return df, trade_orders


def run_backtest_allocation(df, charge, loss_limit, slippage,allocation,multi,mintick):
    """
       执行动态资金分配策略的回测过程。

       参数：
       df : pd.DataFrame
           包含价格数据与交易信号的DataFrame，必须包括列 ['OPEN', 'CLOSE', 'signal', 'CLOCK']。

       charge : float
           每次交易收取的手续费率。

       loss_limit : float
           单日止损阈值，当策略当日收益小于该值时强制平仓。

       slippage : float
           滑点比例，用于模拟成交价与理想价之间的偏差。

       allocation : float
           初始分配资金，即账户初始本金。

       multi : int
           合约乘数，每跳变动对应盈亏金额。

       mintick : float
           最小变动价位，结合乘数用于计算滑点损失。

       返回：
       df : pd.DataFrame
           包含回测结果的原始数据（含收益、仓位、交易行为等）。

       trade_orders : pd.DataFrame
           策略发生交易的订单记录（时间、价格、交易类型等）。
       """
    # ========== 仓位与初始策略收益 ==========
    # 根据昨日信号分配资金，按当日开盘价建仓：allocation / OPEN 为买入手数
    df["position"]=allocation*df["signal"].shift(1).fillna(0)/df["OPEN"]
    df["return"] = df["OPEN"].pct_change().fillna(0)   # 计算每日收益率（基于开盘价）
    df["strategy_return"] = df["return"] * df["position"]  # 策略的每日收益

    # 交易滑点成本（直接使用滑点比例）
    df["trade"] = df["position"].diff().abs()  # 计算每日的交易数量
    df["slip_cost"] = df["trade"] * slippage * multi * mintick  # 计算滑点成本
    df["fee"] = df["trade"] * charge  # 手续费
    df["strategy_return"] -= (df["fee"] + df["slip_cost"])  # 扣除手续费与滑点

    # 记录买入/卖出的类型
    df["order_type"] = None  # 初始化为None
    df.loc[(df["position"] == 1) & (df["position"].shift(1) != 1), "order_type"] = "buy"  # 买入
    df.loc[(df["position"] == 0) & (df["position"].shift(1) == 1), "order_type"] = "sell"  # 卖出
    df.loc[(df["position"] == 0) & (df["position"].shift(1) == -1), "order_type"] = "buy_to_cover"  # 平空操作

    # **执行止损操作，并记录交易**
    # 止损：如果策略的收益小于止损阈值，清仓（即仓位置为0）
    df.loc[df["strategy_return"] < loss_limit, "position"] = 0

    # **止损后记录卖出**
    # 如果止损操作发生（仓位被清除），记录为 "sell" 操作
    df.loc[(df["position"] == 0) & (df["position"].shift(1) == 1), "order_type"] = "sell"

    df["trade"] = df["position"].diff().abs()

    # 止损后重新计算策略的收益
    df["strategy_return"] = df["return"] * df["position"]

    # 累计收益计算
    df["cum_return"] = (1 + df["strategy_return"]).cumprod()  # 策略累计收益
    df["interPNL"] = allocation * df["strategy_return"]  # 每日盈亏
    df["balance"] = allocation + df["interPNL"].cumsum()  # 账户余额

    # 保存每日盈亏和账户余额数据到 CSV 文件
    pnl_data = df[["CLOCK", "interPNL", "balance"]]
    pnl_data.to_csv("pnl.csv", index=False, encoding="utf-8")

    # 生成交易订单列表
    trade_orders = df.loc[df["order_type"].notna(), ["CLOCK", "CLOSE", "order_type", "position", "trade"]]

    # 返回更新后的DataFrame以及生成的交易订单列表
    return df, trade_orders


def calculate_average_holding_period(df):
    """
        计算策略的平均持仓周期（单位：天）。

        参数：
        df : pd.DataFrame
            回测结果数据框，需包含列 ['position', 'CLOCK']。
            'position' 表示每日持仓方向（1：多头，-1：空头，0：空仓）；
            'CLOCK' 为时间戳列，支持被 pd.to_datetime 解析。

        返回：
        avg_holding_period : float
            平均持有天数（开仓到平仓的平均周期），若无交易返回 0。
        """
    holding_periods = []

    # 循环遍历DataFrame
    entry_time = None  # 记录持仓的起始时间
    for i in range(1, len(df)):
        # 当仓位从0变为1时（开多仓），记录开仓时间
        if df["position"].iloc[i - 1] == 0 and df["position"].iloc[i] == 1:
            entry_time = df["CLOCK"].iloc[i]

        # 当仓位从0变为-1时（开空仓），记录开仓时间
        elif df["position"].iloc[i - 1] == 0 and df["position"].iloc[i] == -1:
            entry_time = df["CLOCK"].iloc[i]

        # 当仓位从1变为0时（平多仓），记录平仓时间，并计算持仓周期
        elif df["position"].iloc[i - 1] == 1 and df["position"].iloc[i] == 0 and entry_time is not None:
            exit_time = df["CLOCK"].iloc[i]
            holding_period = (pd.to_datetime(exit_time) - pd.to_datetime(entry_time)).days
            holding_periods.append(holding_period)
            entry_time = None  # 重置开仓时间

        # 当仓位从-1变为0时（平空仓），记录平仓时间，并计算持仓周期
        elif df["position"].iloc[i - 1] == -1 and df["position"].iloc[i] == 0 and entry_time is not None:
            exit_time = df["CLOCK"].iloc[i]
            holding_period = (pd.to_datetime(exit_time) - pd.to_datetime(entry_time)).days
            holding_periods.append(holding_period)
            entry_time = None  # 重置开仓时间

    # 计算平均持有周期
    avg_holding_period = sum(holding_periods) / len(holding_periods) if holding_periods else 0
    return avg_holding_period


def signal_quality_evaluation(df):
    """
        策略信号质量评估函数。
        基于回测数据，评估信号的胜率、盈亏比、最大回撤、波动率和盈亏因子。

        参数：
        df : pd.DataFrame
            包含回测结果的数据框，要求至少包含 'strategy_return' 列。

        返回：
        dict
            包含以下信号质量指标：
            - Win Rate（胜率 %）
            - Profit-to-Loss Ratio（盈亏比）
            - Max Drawdown（最大回撤）
            - Volatility（策略收益波动率）
            - Profit Factor（盈亏因子）
        """
    # 计算信号胜率
    winning_trades = df[df["strategy_return"] > 0]
    total_trades = df[df["strategy_return"].notna()]
    win_rate = len(winning_trades) / len(total_trades) * 100  # 信号胜率百分比

    # 计算盈亏比
    avg_win = winning_trades["strategy_return"].mean()  # 平均盈利
    avg_loss = df[df["strategy_return"] < 0]["strategy_return"].mean()  # 平均亏损
    profit_to_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else None  # 盈亏比

    # 计算最大回撤
    df["cumulative_return"] = (1 + df["strategy_return"]).cumprod()
    max_drawdown = (df["cumulative_return"].max() - df["cumulative_return"].min()) / df["cumulative_return"].max()

    # 计算收益波动率
    volatility = df["strategy_return"].std()  # 收益的标准差

    # 计算盈亏比例
    total_profit = winning_trades["strategy_return"].sum()
    total_loss = df[df["strategy_return"] < 0]["strategy_return"].sum()
    profit_factor = total_profit / abs(total_loss) if total_loss != 0 else None

    return {
        "Win Rate": win_rate,
        "Profit-to-Loss Ratio": profit_to_loss_ratio,
        "Max Drawdown": max_drawdown,
        "Volatility": volatility,
        "Profit Factor": profit_factor
    }


def calculate_net_value(df, allocation):
    """
    账户净值计算函数。

    根据每日持仓情况及市场价格，动态更新账户现金、持仓市值和总净值。
    适用于回测中根据逐日仓位变化计算账户表现。

    参数：
    df : pd.DataFrame
        包含回测数据的DataFrame，需包括'CLOSE'（收盘价）、'position'（持仓量）列，
        可选包含'fee'（手续费）、'slip_cost'（滑点成本）列。
    allocation : float
        初始账户总资金。

    返回：
    df : pd.DataFrame
        添加以下字段：
        - cash：每日剩余现金
        - market_value：每日持仓市值
        - net_value：每日账户净值（现金 + 市值）
        - unit_net_value：单位净值（以初始净值为1）
    """

    # 初始现金为总资金 allocation，其他列表用于记录每日的账户状态
    cash = allocation
    cash_list = []
    market_value_list = []
    net_value_list = []

    prev_position = 0  # 记录上一个交易日的持仓，用于计算增减变化

    for i in range(len(df)):
        price = df.loc[i, "CLOSE"]               # 当日收盘价
        position = df.loc[i, "position"]         # 当前持仓量
        delta = position - prev_position         # 持仓变化（买入/卖出量）

        # 获取手续费和滑点成本，若缺失则默认为0
        fee = df.loc[i, "fee"] if "fee" in df.columns and pd.notna(df.loc[i, "fee"]) else 0
        slip = df.loc[i, "slip_cost"] if "slip_cost" in df.columns and pd.notna(df.loc[i, "slip_cost"]) else 0

        # 本次交易成本 = 持仓变动金额 + 手续费 + 滑点
        trade_cost = delta * price + fee + slip
        cash -= trade_cost                           # 账户现金减少交易成本

        market_value = position * price              # 当前市值 = 持仓量 * 收盘价
        net_value = cash + market_value              # 总账户净值 = 现金 + 市值

        # 记录每日数据
        cash_list.append(cash)
        market_value_list.append(market_value)
        net_value_list.append(net_value)

        prev_position = position                     # 更新前一日持仓

    # 将计算结果写入 DataFrame
    df["cash"] = cash_list
    df["market_value"] = market_value_list
    df["net_value"] = net_value_list
    df["unit_net_value"] = df["net_value"] / df["net_value"].iloc[0]  # 单位净值（标准化）

    return df


def calculate_strategy_performance(df):
    # 计算每日净值变化
    df['unit_net_value'] = df['net_value'] / df['net_value'].iloc[0]

    # 计算日收益率
    df['daily_return'] = df['unit_net_value'].pct_change().fillna(0)

    # 计算年化收益率（假设一年252个交易日）
    annual_return = (1 + df['daily_return'].mean()) ** 252 - 1

    # 计算标准差（年化）
    stddev = df['daily_return'].std() * np.sqrt(252)

    # 计算夏普比率
    sharpe_ratio = df['daily_return'].mean() / df['daily_return'].std() * np.sqrt(252)

    # 计算最大回撤
    cumulative_return = (1 + df['daily_return']).cumprod()
    rolling_max = cumulative_return.cummax()
    drawdown = (cumulative_return - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # 计算卡玛比（Calmar Ratio） = 年化收益 / 最大回撤
    calmar_ratio = annual_return / abs(max_drawdown)

    # 计算绝对收益
    abs_return = df['unit_net_value'].iloc[-1] - 1

    # 返回计算的指标
    backtest = {
        'sharpe': sharpe_ratio,
        'stddev': stddev,
        'calmar': calmar_ratio,
        'AbsReturn': abs_return,
        'AnnualReturn': annual_return,
        'MaxDrawdown': max_drawdown
    }

    return backtest


# 计算策略性能并保存结果
def save_backtest_results(symbol,backtest, begin_date, end_date, slippage, init_cap, file_name="backtest.csv"):
    """
        保存回测结果到CSV文件，用于记录不同参数或策略下的绩效表现。

        参数:
        symbol : str
            标的代码或名。
        backtest : dict
            策略绩效指标的字典，包含如 Sharpe、Calmar、最大回撤等。
        begin_date : str
            回测开始日期，格式如 "2021-01-01"。
        end_date : str
            回测结束日期，格式如 "2021-12-31"。
        slippage : float
            回测使用的滑点值。
        init_cap : float
            初始资金（分配的起始资金）。
        file_name : str
            要保存的CSV文件名，默认为 "backtest.csv"。
        """
    # 创建一个包含回测结果的 DataFrame
    backtest_data = {
        'symbol':[symbol],
        'begin': [begin_date],
        'end': [end_date],
        'slippage': [slippage],
        'initCap': [init_cap],
        'stddev': [backtest['stddev']],
        'sharpe': [backtest['sharpe']],
        'calmar': [backtest['calmar']],
        'AbsReturn': [backtest['AbsReturn']],
        'AnnualReturn': [backtest['AnnualReturn']],
        'MaxDrawdown': [backtest['MaxDrawdown']],
        'DataVersion': ['v1']
    }

    # 转换为 DataFrame
    df_backtest = pd.DataFrame(backtest_data)

    # 保存到 CSV 文件
    df_backtest.to_csv(file_name, index=False, mode='a', header=not pd.io.common.file_exists(file_name))


def plot_signals(df, windows_1, windows_2,fileName="plot_signals.png"):
    """
       绘制买入/卖出信号图表
       参数:
       df : pandas.DataFrame
           包含时间、收盘价、均线、交易信号的数据框，需包含 'CLOCK', 'CLOSE', 'ma1', 'ma2', 'signal' 列。
       windows_1 : int
           短期均线的窗口大小（用于图例显示）。
       windows_2 : int
           长期均线的窗口大小（用于图例显示）。
       fileName : str
           保存图片的文件名，默认为 "plot_signals.png"。
       """
    plt.figure(figsize=(14, 6))
    plt.plot(df["CLOCK"], df["CLOSE"], label="Close", color='black', linewidth=1)
    plt.plot(df["CLOCK"], df["ma1"], label=f"MA{windows_1}", color='blue', linestyle='--')
    plt.plot(df["CLOCK"], df["ma2"], label=f"MA{windows_2}", color='orange', linestyle='--')

    # 画出买入信号：signal == 1
    buy_signals = df[df["signal"] == 1]
    plt.scatter(buy_signals["CLOCK"], buy_signals["CLOSE"], marker="^", color="green", label="Buy Signal", s=60)

    # 画出卖出信号：signal == -1
    sell_signals = df[df["signal"] == -1]
    plt.scatter(sell_signals["CLOCK"], sell_signals["CLOSE"], marker="v", color="red", label="Sell Signal", s=60)

    plt.title("Buy/Sell Signal Chart")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.xticks(df["CLOCK"][::60], rotation=45)
    yticks = np.arange(1000, 15001, 1000)
    plt.yticks(yticks)
    plt.ylim(1000, 15000)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(fileName)
    plt.show()
    plt.close()


def plot_position(df, fileName="position_simulation_line.png"):
    """
        绘制持仓情况图，包括价格走势、持仓变化和账户净值（双 Y 轴）

        参数:
        df : pandas.DataFrame
            包含 'CLOCK', 'CLOSE', 'position', 'balance' 等列的数据。
        fileName : str
            图片保存的文件名，默认为 "position_simulation_line.png"
        """
    fig, ax1 = plt.subplots(figsize=(12, 6))  # 创建主图和主y轴

    # 绘制 close price（左 y 轴）

    # 绘制基准线（position = 0）
    ax1.axhline(6000, color='black', linewidth=1)  # 添加横向基准线

    # 绘制 position 的竖条（绿色表示多头，红色表示空头，灰色表示无仓位）
    ax1.plot(df["CLOCK"], (df["position"] * 6000) + 6000, label="Position", color="green", linestyle='-', linewidth=2)

    ax1.plot(df["CLOCK"], df["CLOSE"], label="close price", color="b")
    ax1.set_xlabel("time")
    ax1.set_ylabel("close price", color="b")
    ax1.tick_params(axis='y', labelcolor="b")
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))



    # 绘制账户净值 balance（右 y 轴）
    ax2 = ax1.twinx()
    ax2.plot(df["CLOCK"], df["balance"], label="balance", color="purple", linestyle="--")
    ax2.set_ylabel("balance", color="purple")
    ax2.tick_params(axis='y', labelcolor="purple")
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))

    # 设置时间 x 轴刻度
    ax1.set_xticks(df["CLOCK"][::60])
    ax1.set_xticklabels(df["CLOCK"][::60], rotation=45, ha='right')

    # 图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    fig.tight_layout()
    plt.savefig(fileName)
    plt.show()
    plt.close()


def plot_netValue(df, fileName="plot_netValue.png"):
    """
    绘制单位净值曲线图

    参数:
    df : pandas.DataFrame
        包含 'CLOCK'（时间戳）和 'unit_net_value'（单位净值）列的数据。
    fileName : str
        图表保存的文件名，默认为 "plot_netValue.png"
    """
    plt.figure(figsize=(12, 4))
    plt.plot(df["CLOCK"], df["unit_net_value"], label="unit_net_value", color='blue')
    plt.title("unite net Value curve")
    plt.xlabel("time")
    plt.ylabel("unit_net_value")
    plt.xticks(df["CLOCK"][::60], rotation=45)
    plt.yticks(df["unit_net_value"][::40], rotation=45)
    ymin = df["unit_net_value"].min()
    ymax = df["unit_net_value"].max()
    yticks = np.linspace(ymin, ymax, 10)
    plt.yticks(yticks, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fileName)
    plt.show()
    plt.close()


def plot_pnl(df, allocation, fileName="plot_pnl.png"):
    """
       绘制策略累计收益和账户净值曲线（双Y轴）

       参数:
       df : pandas.DataFrame
           包含 'CLOCK', 'cum_return', 'balance' 等列的回测结果数据。
       allocation : float
           初始分配资金，用于归一化账户净值。
       fileName : str
           图表保存的文件名，默认为 "plot_pnl.png"
       """
    fig, ax1 = plt.subplots(figsize=(12, 6))  # 创建一个图表和主y轴

    # 在主y轴上绘制累计收益曲线
    ax1.plot(df["CLOCK"], df["cum_return"], label="Cumulative strategy returns", color="b")
    ax1.set_xlabel("time")
    ax1.set_ylabel("Cumulative strategy returns (%)", color="b")
    ax1.tick_params(axis='y', labelcolor="b")  # 设置主y轴的刻度颜色为蓝色
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))

    # 创建第二个y轴来绘制账户净值曲线
    ax2 = ax1.twinx()  # 共享x轴，创建一个新的y轴
    ax2.plot(df["balance"] / allocation, label="Account Net Value", color="g", linestyle="--")
    ax2.set_ylabel("Account Net Value", color="g")
    ax2.tick_params(axis='y', labelcolor="g")  # 设置第二个y轴的刻度颜色为绿色
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))

    # 设置标题和图例
    ax1.set_xticks(df["CLOCK"][::60])  # 每60个数据点设置一个刻度
    ax1.set_xticklabels(df["CLOCK"][::60], rotation=45, ha='right')  # 格式化日期并旋转
    # 自动调整布局，以防止标签重叠
    fig.tight_layout()

    # 保存并显示图表
    plt.savefig(fileName)
    plt.show()
    plt.close()

def plot_pnl_compare(df,df1,df3, allocation, fileName="plot_pnl_compare.png"):
    """
        对比绘制多个策略的累计收益曲线和账户净值曲线（双Y轴）

        参数:
        df, df1, df3 : pandas.DataFrame
            包含 'CLOCK', 'cum_return', 'balance' 等列的数据，分别对应不同策略。
        allocation : float
            初始分配资金，用于归一化账户净值。
        fileName : str
            图表保存的文件名，默认为 "plot_pnl_compare.png"
        """
    fig, ax1 = plt.subplots(figsize=(12, 6))  # 创建一个图表和主y轴

    # 在主y轴上绘制累计收益曲线
    ax1.plot(df["CLOCK"], df["cum_return"], label="Cumulative strategy returns", color="blue")
    ax1.plot(df1["CLOCK"], df1["cum_return"], label="Cumulative strategy1 returns", color="red")
    ax1.plot(df3["CLOCK"], df3["cum_return"]/500, label="Cumulative strategy2 returns", color="yellow")
    ax1.set_xlabel("time")
    ax1.set_ylabel("Cumulative strategy returns (%)", color="b")
    ax1.tick_params(axis='y', labelcolor="black")  # 设置主y轴的刻度颜色为蓝色
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))

    # 创建第二个y轴来绘制账户净值曲线
    ax2 = ax1.twinx()  # 共享x轴，创建一个新的y轴
    ax2.plot(df["balance"] / allocation, label="Account Net Value", color="blue", linestyle="--")
    ax2.plot(df1["balance"] / allocation, label="Account Net Value1", color="red", linestyle="--")
    ax2.plot(df3["balance"] / (allocation*500), label="Account Net Value2", color="yellow", linestyle="--")
    ax2.set_ylabel("Account Net Value", color="g")
    ax2.tick_params(axis='y', labelcolor="black")  # 设置第二个y轴的刻度颜色为绿色
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))

    # 设置标题和图例
    ax1.set_xticks(df["CLOCK"][::60])  # 每60个数据点设置一个刻度
    ax1.set_xticklabels(df["CLOCK"][::60], rotation=45, ha='right')  # 格式化日期并旋转

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize="medium")

    # 自动调整布局，以防止标签重叠
    fig.tight_layout()
    plt.title("Strategy Comparison: Return and Net Value")



    # 保存并显示图表
    plt.savefig(fileName)
    plt.show()
    plt.close()


# ====== 参数敏感性分析 ======
def sensitivity_analysis(df, short_range, long_range, fileName="sensitivity_heatmap.png"):
    """
        策略参数敏感性分析：遍历不同短期和长期均线组合，评估策略性能（以Sharpe比率为例），并绘制热力图。

        参数:
        df : pandas.DataFrame
            原始行情数据，包含CLOSE价格、时间等。
        short_range : list
            短期均线的参数范围（如range(5, 30, 5)）。
        long_range : list
            长期均线的参数范围（如range(40, 120, 20)）。
        fileName : str
            热力图保存的文件名，默认为 "sensitivity_heatmap.png"。
        """
    performance_metrics = []

    # 遍历不同的短期和长期窗口
    for short_window in short_range:
        for long_window in long_range:
            # 生成信号并进行回测
            temp_df = generate_signals(df.copy(), short_window, long_window)
            temp_df, trade_orders = run_backtest(temp_df, charge, loss_limit, slippage, allocation, multi, mintick)
            temp_df = calculate_net_value(temp_df, allocation)
            backtest = calculate_strategy_performance(temp_df)
            save_backtest_results(backtest, begin, end, slippage, allocation)

            # 记录每个参数组合的回测表现
            performance_metrics.append({
                'short_window': short_window,
                'long_window': long_window,
                'sharpe': backtest['sharpe'],
                'max_drawdown': backtest['MaxDrawdown'],
                'annual_return': backtest['AnnualReturn'],
                'calmar': backtest['calmar'],
                'abs_return': backtest['AbsReturn']
            })

    # 将结果转换为DataFrame
    metrics_df = pd.DataFrame(performance_metrics)

    # 将绩效指标透视成适合做热力图的格式
    heatmap_data = metrics_df.pivot(index='short_window', columns='long_window', values='sharpe')  # 夏普比率

    # 绘制热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".2f", cbar_kws={'label': 'Sharpe Ratio'})
    plt.title("Sensitivity Analysis Heatmap: Sharpe Ratio")
    plt.xlabel("Long Window")
    plt.ylabel("Short Window")
    plt.tight_layout()
    plt.savefig(fileName)
    plt.show()

def main():
    for symbol in symbols:
        df = load_single_csv("day_20220611",symbol)
        df = generate_signals(df, windows_1, windows_2)
        df,trade_orders = run_backtest_allocation(df, charge, loss_limit,slippage,allocation,multi,mintick)
        #df.to_csv('tradeOrderanalysis.csv', index=False, encoding='utf-8')
        signal_quality = signal_quality_evaluation(df)
        print(signal_quality)
        df=calculate_net_value(df,allocation)
        #trade_orders.to_csv('trade_orders.csv', index=False, encoding='utf-8')
        #print(trade_orders)
        backtest = calculate_strategy_performance(df)
        save_backtest_results(symbol,backtest, begin, end, slippage, allocation)
        print("策略评估指标：", backtest)


        #plot_signals(df, windows_3, windows_4)
        #plot_position(df)
        #plot_netValue(df)
        #plot_pnl(df,allocation)

        #sensitivity_analysis(df, short_range, long_range)


if __name__ == "__main__":
    data=main()


