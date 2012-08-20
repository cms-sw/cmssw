timingClient = dict(
    expectedMean = 0.,
    meanThreshold = 2.,
    rmsThreshold = 6.,
    minChannelEntries = 3,
    minTowerEntries = 10,
    tailPopulThreshold = 0.3,
    sources = dict(
        TimeAllMap = ['TimingTask', 'TimeAllMap'],
        TimeMap = ['TimingTask', 'TimeMap']
    )
)

timingClientPaths = dict(
    Quality = "Timing/Quality/TimingClient timing quality",
    MeanSM = "Timing/Mean/TimingClient SM mean",
    MeanAll = "Timing/TimingClient timing mean",
    FwdBkwdDiff = "Timing/TimingClient forward - backward",
    FwdvBkwd = "Timing/TimingClient forward v backward",
    RMS = "Timing/RMS/TimingClient SM rms",
    RMSAll = "Timing/TimingClient timing RMS",
    Projection = "Timing/TimingClient timing projection",
    QualitySummary = "Summary/TimingClient timing quality"
)
