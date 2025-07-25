#feature select
def _xgboost_rfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    savePath=None,
    dpi=600,
    picFormat="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    model: XGBOOST模型，如果不传则自动产生一个自动寻参后的XGBOOST模型
    searching: 是否自动寻参，默认为是
    savePath:str 图片存储路径

    hyperparams: XGBClassifier params -- no selection yet
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("Regression", XGBRegressor())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(XGBRegressor(), param_distributions=GridDefaultRange['XGBRegressor'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = XGBRegressor(**model.best_params_)
    else:
        model = XGBRegressor(random_state=42)
    # if searching:
    #     str_result = "采用XGBoost进行变量重要度分析，模型参数为:\n" + dic2str(model.best_params_, 'XGBRegressor')
    # else:
    str_result = "算法：XGBoost回归模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用XGBoost进行变量重要度分析，模型参数为:\n" + dic2str(model.get_params(), model.__class__.__name__)
    model.fit(x, y)

    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": abs(model.feature_importances_)}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n".format(
        top_features, str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if savePath is not None:
        plot_name_dict = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            savePath,
            dpi=dpi,
            picFormat=picFormat,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]



def _randomforest_rfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    savePath=None,
    dpi=600,
    picFormat="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    num_features:图表中展示的特征数量
    (会剔除空值)
    savePath:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("Regression", RandomForestRegressor())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(RandomForestRegressor(), param_distributions=GridDefaultRange['RandomForestRegressor'],random_state=42, )
        #searcher = BayesSearchCV(RandomForestRegressor(),  search_spaces=BayesDefaultRange['RandomForestRegressor']  # 使用字符串键)
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = RandomForestRegressor(**model.best_params_).fit(x, y)
    else:
        model = RandomForestRegressor(random_state=42).fit(x, y)
    param_dict = model.get_params()

    # if searching:
    #     str_result = "采用Random Forrest Regressor进行变量重要度分析，模型参数为:\n" + dic2str(model.best_params_, 'RandomForestRegressor')
    # else:
    str_result = "算法：随机森林回归模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用Random Forrest Regressor进行变量重要度分析，模型参数为:\n" + dic2str(
            param_dict, model.__class__.__name__)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": abs(model.feature_importances_)}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n".format(
        top_features, str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if savePath is not None:
        plot_name_dict = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            savePath,
            dpi=dpi,
            picFormat=picFormat,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]


def _adaboost_rfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    savePath=None,
    dpi=600,
    picFormat="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    num_features:图表中展示的特征数量
    (会剔除空值)
    savePath:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("Regression", AdaBoostRegressor())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(AdaBoostRegressor(), param_distributions=GridDefaultRange['AdaBoostRegressor'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = AdaBoostRegressor(**model.best_params_).fit(x, y)
    else:
        model = AdaBoostRegressor(random_state=42).fit(x, y)
    param_dict = model.get_params()

    # if searching:
    #     str_result = "采用AdaBoost Regressor进行变量重要度分析，模型参数为:\n" + dic2str(model.best_params_, 'AdaBoostRegressor')
    # else:
    str_result = "算法：AdaBoost回归模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用AdaBoost Regressor进行变量重要度分析，模型参数为:\n" + dic2str(
            param_dict, model.__class__.__name__)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": abs(model.feature_importances_)}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n".format(
        top_features, str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if savePath is not None:
        plot_name_dict = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            savePath,
            dpi=dpi,
            picFormat=picFormat,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]


def _linear_rfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    savePath=None,
    dpi=600,
    picFormat="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    num_features:图表中展示的特征数量
    (会剔除空值)
    savePath:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)
    #获取模型参数
    param_dict = model.get_params()
    #将模型参数转化为字符串
    str_result = "算法：线性回归模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用Linear Regression进行变量重要度分析，模型参数为:\n" + dic2str(
            param_dict, model.__class__.__name__)
    #获取系数并排序
    df_result = pd.DataFrame({
        "Variable": x_columns,
        "Weight Importance": abs(model.coef_)
    }).sort_values(by="Weight Importance", ascending=False)
    #获取重要度最高的特征
    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n".format(
        top_features, str(top_list)[1:-1]
    )
    str_result += "\n注意：在使用线性回归进行重要度排序时，由于其线性特性，将默认所有变量具有相同量纲和值域。这一假设在勾选数据标准化后可认为成立。如果线性模型(与常识对比)在此数据集上表现较差，可考虑使用非线性模型如XGBoost。"

    plot_name_dict = {}
    #绘制图形
    if savePath is not None:
        plot_name_dict = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            savePath,
            dpi=dpi,
            picFormat=picFormat,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]

def _kneighb_rfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    savePath=None,
    dpi=600,
    picFormat="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    num_features:图表中展示的特征数量
    (会剔除空值)
    savePath:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna().sample(frac=1, random_state=42).reset_index(drop=True)
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("classification", KNeighborsRegressor())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(KNeighborsRegressor(),
                                      param_distributions=GridDefaultRange['KNeighborsRegressor'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = KNeighborsRegressor(**model.best_params_).fit(x, y)
    else:
        model = KNeighborsRegressor().fit(x, y)
    #获取模型参数
    param_dict = model.get_params()
    #将模型参数转化为字符串
    str_result = "算法：K近邻回归模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用K近邻回归进行变量重要度分析，模型参数为:\n" + dic2str(
            param_dict, model.__class__.__name__)
    #获取系数并排序
    weight_im = abs(
        permutation_importance(
            model, x, y, n_repeats=10, random_state=0
        ).importances_mean
    )
    weight_im = weight_im / sum(weight_im)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": weight_im}
    ).sort_values(by="Weight Importance", ascending=False)

    #获取重要度最高的特征
    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n".format(
        top_features, str(top_list)[1:-1]
    )

    plot_name_dict = {}
    #绘制图形
    if savePath is not None:
        plot_name_dict = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            savePath,
            dpi=dpi,
            picFormat=picFormat,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]

def _LinearSVM_rfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    savePath=None,
    dpi=600,
    picFormat="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    num_features:图表中展示的特征数量
    (会剔除空值)
    savePath:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("classification", LinearSVR())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(LinearSVR(), param_distributions=GridDefaultRange['LinearSVR'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = LinearSVR(**model.best_params_).fit(x, y)
    else:
        model = LinearSVR(random_state=42).fit(x, y)
    param_dict = model.get_params()
    #将模型参数转化为字符串
    str_result = "算法：线性支持向量机回归模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用线性支持向量机分类算法进行变量重要度分析，模型参数为:\n" + dic2str(
        param_dict, model.__class__.__name__)
    # if searching:
    #     str_result = "采用支持向量机分类算法进行变量重要度分析，模型参数为:\n" + dic2str(model.best_params_, 'LinearSVR')
    # else:
    #     str_result = "采用支持向量机分类算法进行变量重要度分析，模型参数为:\n" + dic2str(
    #         param_dict, model.__class__.__name__)
    #排序
    weight_im = abs(
        permutation_importance(
            model, x, y, n_repeats=10, random_state=0
        ).importances_mean
    )
    weight_im = weight_im / sum(weight_im)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": weight_im}
    ).sort_values(by="Weight Importance", ascending=False)
    #获取重要度最高的特征
    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n".format(
        top_features, str(top_list)[1:-1]
    )
    #画图
    plot_name_dict = {}
    if savePath is not None:
        plot_name_dict = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            savePath,
            dpi=dpi,
            picFormat=picFormat,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]



# ----------分类重要度排序-----------
def _logisticL1_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    savePath=None,
    dpi=600,
    picFormat="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    num_features:图表中展示的特征数量
    (会剔除空值)
    savePath:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]
    crange = np.logspace(-9, 1, 200, base=10)

    logicv_ = LogisticRegressionCV(Cs=crange, cv=5, penalty="l1", solver="saga",random_state=42).fit(
        x, y
    )
    param_dict = logicv_.get_params()
    param_dict["C"] = logicv_.C_
    str_result = "算法：逻辑回归分类模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用L1正则化的Logistic回归进行变量重要度分析，模型参数为:\n" + dic2str(
        param_dict, logicv_.__class__.__name__
    )

    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": abs(logicv_.coef_[0])}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n".format(
        top_features, str(top_list)[1:-1]
    )
    str_result += "\n注意：在使用Logistiv+L1进行重要度排序时，由于其指数部分的线性形式，将默认所有变量具有相同量纲和值域。这一假设在勾选数据标准化后可认为成立。如果线性模型(与常识对比)在此数据集上表现较差，可考虑使用非线性模型如XGBoost。"

    plot_name_dict = {}
    if savePath is not None:
        plot_name_dict = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            savePath,
            dpi=dpi,
            picFormat=picFormat,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]

def _logistic2_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    savePath=None,
    dpi=600,
    picFormat="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    num_features:图表中展示的特征数量
    (会剔除空值)
    savePath:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]
    # 对因变量进行有序编码
    encoder = OrdinalEncoder()
    y_encoded = encoder.fit_transform(y.values.reshape(-1, 1)).ravel()

    logicv = LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial', random_state=42).fit(
        x, y_encoded
    )
    param_dict = logicv.get_params()
    param_dict["C"] = logicv.C
    str_result = "算法：有序逻辑回归模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用有序逻辑回归进行变量重要度分析，模型参数为:\n" + dic2str(
        param_dict, logicv.__class__.__name__
    )

    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": abs(logicv.coef_[0])}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n".format(
        top_features, str(top_list)[1:-1]
    )
    str_result += "\n注意：函数对结局变量进行了有序编码。这是有序逻辑回归所必需的。用户需要确保结局变量是有序的分类变量。"
    plot_name_dict = {}
    if savePath is not None:
        plot_name_dict = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            savePath,
            dpi=dpi,
            picFormat=picFormat,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]


def _xgboost_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    savePath=None,
    dpi=600,
    picFormat="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    model: XGBOOST模型，如果不传则自动产生一个自动寻参后的XGBOOST模型
    searching: 是否自动寻参，默认为是
    savePath:str 图片存储路径

    hyperparams: XGBClassifier params -- no selection yet
    """
    x = df_input[x_columns]
    y = df_input[y_column]

    if searching:
        # searcher = RandSearcherCV("Classification", XGBClassifier())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(XGBClassifier(), param_distributions=GridDefaultRange['XGBClassifier'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = XGBClassifier(**model.best_params_).fit(x, y)
    else:
        model = XGBClassifier(random_state=42).fit(x, y)
    # if searching:
    #     str_result = "采用极端梯度提升树(XGBOOST)进行变量重要度分析，模型参数为:\n" + dic2str(model.best_params_, 'XGBClassifier')
    # else:
    str_result = "算法：XGBoost分类模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用极端梯度提升树(XGBOOST)进行变量重要度分析，模型参数为:\n" + dic2str(
            model.get_params(), model.__class__.__name__)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": abs(model.feature_importances_)}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n".format(
        top_features, str(top_list)[1:-1]
    )
    plot_name_dict = {}
    if savePath is not None:
        plot_name_dict = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            savePath,
            dpi=dpi,
            picFormat=picFormat,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]


def _randomforest_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    savePath=None,
    dpi=600,
    picFormat="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    num_features:图表中展示的特征数量
    (会剔除空值)
    savePath:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("Classification", RandomForestClassifier())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(RandomForestClassifier(),param_distributions=GridDefaultRange['RandomForestClassifier'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = RandomForestClassifier(**model.best_params_).fit(x, y)
    else:
        model = RandomForestClassifier(random_state=42).fit(x, y)
    param_dict = model.get_params()


    # if searching:
    #     str_result = "采用Random Forrest Classifier进行变量重要度分析，模型参数为:\n" + dic2str(model.best_params_, 'RandomForestClassifier')
    # else:
    str_result = "算法：随机森林模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用Random Forrest Classifier进行变量重要度分析，模型参数为:\n" + dic2str(
            param_dict, model.__class__.__name__)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": abs(model.feature_importances_)}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n".format(
        top_features, str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if savePath is not None:
        plot_name_dict = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            savePath,
            dpi=dpi,
            picFormat=picFormat,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]


def _adaboost_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    savePath=None,
    dpi=600,
    picFormat="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    num_features:图表中展示的特征数量
    (会剔除空值)
    savePath:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("classification", AdaBoostClassifier())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(AdaBoostClassifier(), param_distributions=GridDefaultRange['AdaBoostClassifier'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = AdaBoostClassifier(**model.best_params_).fit(x, y)
    else:
        model = AdaBoostClassifier(random_state=42).fit(x, y)
    param_dict = model.get_params()

    str_result = "算法：AdaBoost分类模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用AdaBoost Classifier进行变量重要度分析，模型参数为:\n" + dic2str(
                param_dict, model.__class__.__name__)


    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": abs(model.feature_importances_)}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n".format(
        top_features, str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if savePath is not None:
        plot_name_dict = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            savePath,
            dpi=dpi,
            picFormat=picFormat,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]

def _DecisionTree_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    savePath=None,
    dpi=600,
    picFormat="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    num_features:图表中展示的特征数量
    (会剔除空值)
    savePath:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:

        searcher = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions=GridDefaultRange['DecisionTreeClassifier'],random_state=42, )
        model = searcher.fit(x, y)
        model = DecisionTreeClassifier(**model.best_params_).fit(x, y)
    else:
        model = DecisionTreeClassifier(random_state=42).fit(x, y)
    param_dict = model.get_params()
    str_result = "算法：决策树分类模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用DecisionTree进行变量重要度分析，模型参数为:\n" + dic2str(
                param_dict, model.__class__.__name__)


    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": abs(model.feature_importances_)}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n".format(
        top_features, str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if savePath is not None:
        plot_name_dict = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            savePath,
            dpi=dpi,
            picFormat=picFormat,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]

def _GBDT_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    savePath=None,
    dpi=600,
    picFormat="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    num_features:图表中展示的特征数量
    (会剔除空值)
    savePath:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        searcher = RandomizedSearchCV(GradientBoostingClassifier(), param_distributions=GridDefaultRange['GradientBoostingClassifier'],random_state=42, )
        model = searcher.fit(x, y)
        model = GradientBoostingClassifier(**model.best_params_).fit(x, y)
    else:
        model = GradientBoostingClassifier(random_state=42).fit(x, y)
    param_dict = model.get_params()
    str_result = "算法：GBDT分类模型\n"
    str_result += "变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用GBDT进行变量重要度分析，模型参数为:\n" + dic2str(
                param_dict, model.__class__.__name__)


    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": abs(model.feature_importances_)}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])

    str_result += "\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n".format(
        top_features, str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if savePath is not None:
        plot_name_dict = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            savePath,
            dpi=dpi,
            picFormat=picFormat,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]

def _gussnb_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    savePath=None,
    dpi=600,
    picFormat="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    num_features:图表中展示的特征数量
    (会剔除空值)
    savePath:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("classification", GaussianNB())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(GaussianNB(), param_distributions=GridDefaultRange['GaussianNB'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = GaussianNB(**model.best_params_).fit(x, y)
    else:
        model = GaussianNB().fit(x, y)
    param_dict = model.get_params()


    # if searching:
    #     str_result = "采用高斯朴素贝叶斯分类算法进行变量重要度分析，模型参数为:\n" + dic2str(model.best_params_, 'GaussianNB')
    # else:
    str_result = "算法：高斯朴素贝叶斯分类模型\n"
    str_result += "变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用高斯朴素贝叶斯分类算法进行变量重要度分析，模型参数为:\n" + dic2str(
            param_dict, model.__class__.__name__)
    weight_im = abs(
        permutation_importance(
            model, x, y, n_repeats=10, random_state=0
        ).importances_mean
    )
    weight_im = weight_im / sum(weight_im)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": weight_im}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n".format(
        top_features, str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if savePath is not None:
        plot_name_dict = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            savePath,
            dpi=dpi,
            picFormat=picFormat,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]


def _cnb_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    savePath=None,
    dpi=600,
    picFormat="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    num_features:图表中展示的特征数量
    (会剔除空值)
    savePath:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("classification", ComplementNB())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(ComplementNB(), param_distributions=GridDefaultRange['ComplementNB'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = ComplementNB(**model.best_params_).fit(x, y)
    else:
        model = ComplementNB().fit(x, y)
    param_dict = model.get_params()
    str_result = "算法：补朴素贝叶斯分类模型\n"
    str_result += "变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用补朴素贝叶斯分类算法进行变量重要度分析，模型参数为:\n" + dic2str(
        param_dict, model.__class__.__name__)
    # if searching:
    #     str_result = "采用补朴素贝叶斯分类算法进行变量重要度分析，模型参数为:\n" + dic2str(model.best_params_, 'ComplementNB')
    # else:
    #     str_result = "采用补朴素贝叶斯分类算法进行变量重要度分析，模型参数为:\n" + dic2str(
    #         param_dict, model.__class__.__name__)
    weight_im = abs(
        permutation_importance(
            model, x, y, n_repeats=10, random_state=0
        ).importances_mean
    )
    weight_im = weight_im / sum(weight_im)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": weight_im}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n".format(
        top_features, str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if savePath is not None:
        plot_name_dict = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            savePath,
            dpi=dpi,
            picFormat=picFormat,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]


def _mlp_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    savePath=None,
    dpi=600,
    picFormat="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    num_features:图表中展示的特征数量
    (会剔除空值)
    savePath:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("classification", MLPClassifier())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(MLPClassifier(), param_distributions=GridDefaultRange['MLPClassifier'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = MLPClassifier(**model.best_params_).fit(x, y)
    else:
        model = MLPClassifier(random_state=42).fit(x, y)
    param_dict = model.get_params()
    str_result = "算法：神经网络分类模型\n"
    str_result += "变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用多层感知器（神经网络）分类算法进行变量重要度分析，模型参数为:\n" + dic2str(
        param_dict, model.__class__.__name__)
    # if searching:
    #     str_result = "采用多层感知器（神经网络）分类算法进行变量重要度分析，模型参数为:\n" + dic2str(model.best_params_, 'XGBRegressor')
    # else:
    #     str_result = "采用多层感知器（神经网络）分类算法进行变量重要度分析，模型参数为:\n" + dic2str(
    #         param_dict, model.__class__.__name__)
    weight_im = abs(
        permutation_importance(
            model, x, y, n_repeats=10, random_state=0
        ).importances_mean
    )
    weight_im = weight_im / sum(weight_im)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": weight_im}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n".format(
        top_features, str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if savePath is not None:
        plot_name_dict = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            savePath,
            dpi=dpi,
            picFormat=picFormat,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]


def _svm_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    savePath=None,
    dpi=600,
    picFormat="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    num_features:图表中展示的特征数量
    (会剔除空值)
    savePath:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("classification", SVC())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(SVC(), param_distributions=GridDefaultRange['SVC'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = SVC(**model.best_params_).fit(x, y)
    else:
        model = SVC(random_state=42).fit(x, y)
    param_dict = model.get_params()
    str_result = "算法：支持向量机分类模型\n"
    str_result += "变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用支持向量机分类算法进行变量重要度分析，模型参数为:\n" + dic2str(
        param_dict, model.__class__.__name__)
    # if searching:
    #     str_result = "采用支持向量机分类算法进行变量重要度分析，模型参数为:\n" + dic2str(model.best_params_, 'SVC')
    # else:
    #     str_result = "采用支持向量机分类算法进行变量重要度分析，模型参数为:\n" + dic2str(
    #         param_dict, model.__class__.__name__)
    weight_im = abs(
        permutation_importance(
            model, x, y, n_repeats=10, random_state=0
        ).importances_mean
    )
    weight_im = weight_im / sum(weight_im)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": weight_im}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n".format(
        top_features, str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if savePath is not None:
        plot_name_dict = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            savePath,
            dpi=dpi,
            picFormat=picFormat,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]


def _kneighb_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    savePath=None,
    dpi=600,
    picFormat="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    num_features:图表中展示的特征数量
    (会剔除空值)
    savePath:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("classification", KNeighborsClassifier())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(KNeighborsClassifier(), param_distributions=GridDefaultRange['KNeighborsClassifier'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = KNeighborsClassifier(**model.best_params_).fit(x, y)
    else:
        model = KNeighborsClassifier().fit(x, y)
    param_dict = model.get_params()

    # if searching:
    #     str_result = "采用K近邻分类算法进行变量重要度分析，模型参数为:\n" + dic2str(model.best_params_, 'KNeighborsClassifier')
    # else:
    str_result = "算法：K近邻分类模型\n"
    str_result += "变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用K近邻分类算法进行变量重要度分析，模型参数为:\n" + dic2str(
            param_dict, model.__class__.__name__)
    weight_im = abs(
        permutation_importance(
            model, x, y, n_repeats=10, random_state=0
        ).importances_mean
    )
    weight_im = weight_im / sum(weight_im)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": weight_im}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n".format(
        top_features, str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if savePath is not None:
        plot_name_dict = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            savePath,
            dpi=dpi,
            picFormat=picFormat,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]


def _lightgbm_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    savePath=None,
    dpi=600,
    picFormat="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    num_features:图表中展示的特征数量
    (会剔除空值)
    savePath:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("classification", LGBMClassifier())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(LGBMClassifier(), param_distributions=GridDefaultRange['LGBMClassifier'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = LGBMClassifier(**model.best_params_).fit(x, y)
    else:
        model = LGBMClassifier(random_state=42).fit(x, y)
    param_dict = model.get_params()
    str_result = "算法：LinghtGBM分类模型\n"
    str_result += "变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用LightGBM分类算法进行变量重要度分析，模型参数为:\n" + dic2str(
        param_dict, model.__class__.__name__)
    # if searching:
    #     str_result = "采用LightGBM分类算法进行变量重要度分析，模型参数为:\n" + dic2str(model.best_params_, 'LGBMClassifier')
    # else:
    #     str_result = "采用LightGBM分类算法进行变量重要度分析，模型参数为:\n" + dic2str(
    #         param_dict, model.__class__.__name__)
    weight_im = abs(
        permutation_importance(
            model, x, y, n_repeats=10, random_state=0
        ).importances_mean
    )
    weight_im = weight_im / sum(weight_im)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": weight_im}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的{0}个变量（由高到低）分别为：{1}。\n".format(
        top_features, str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if savePath is not None:
        plot_name_dict = x5.horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            savePath,
            dpi=dpi,
            picFormat=picFormat,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]