from DQM.EcalBarrelMonitorTasks.TimingTask_cfi import timingTask

timingClient = dict(
    expectedMean = 0.,
    toleranceMean = 2.,
    toleranceMeanFwd = 3.,
    toleranceRMS = 6.,
    toleranceRMSFwd = 9.,
    minChannelEntries = 5,
    minTowerEntries = 20,
    tailPopulThreshold = 0.4,
    MEs = dict(
        Quality = dict(path = "Timing/Quality/TimingClient timing quality", otype = 'SM', btype = 'Crystal', kind = 'TH2F'),
        MeanSM = dict(path = "Timing/Mean/TimingClient SM mean", otype = 'SM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': -25., 'high': 25.}),
        MeanAll = dict(path = "Timing/TimingClient timing mean", otype = 'Ecal2P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': -25., 'high': 25.}),
        FwdBkwdDiff = dict(path = "Timing/TimingClient forward - backward", otype = 'Ecal2P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': -5., 'high': 5.}),
        FwdvBkwd = dict(path = "Timing/TimingClient forward v backward", otype = 'Ecal2P', btype = 'User', kind = 'TH2F', xaxis = {'nbins': 50, 'low': -25., 'high': 25.}, yaxis = {'nbins': 50, 'low': -25., 'high': 25.}),
        RMSMap = dict(path = "Timing/RMS/TimingClient SM rms", otype = 'SM', btype = 'Crystal', kind = 'TH2F'),
        RMSAll = dict(path = "Timing/TimingClient timing RMS", otype = 'Ecal2P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 10.}),
        ProjEta = dict(path = "Timing/TimingClient timing projection", otype = 'Ecal3P', btype = 'ProjEta', kind = 'TProfile'),
        ProjPhi = dict(path = "Timing/TimingClient timing projection", otype = 'Ecal3P', btype = 'ProjPhi', kind = 'TProfile'),
        QualitySummary = dict(path = "Summary/TimingClient timing quality", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F')
    ),
    sources = dict(
        TimeAllMap = timingTask['MEs']['TimeAllMap'],
        TimeMap = timingTask['MEs']['TimeMap']
    )
)
