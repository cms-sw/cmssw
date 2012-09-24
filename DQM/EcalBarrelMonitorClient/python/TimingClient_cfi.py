from DQM.EcalBarrelMonitorTasks.TimingTask_cfi import ecalTimingTask

ecalTimingClient = dict(
    expectedMean = 0.,
    toleranceMean = 2.,
    toleranceMeanFwd = 6.,
    toleranceRMS = 6.,
    toleranceRMSFwd = 12.,
    minChannelEntries = 5,
    minChannelEntriesFwd = 40,
    minTowerEntries = 15,
    minTowerEntriesFwd = 160,
    tailPopulThreshold = 0.4,
    MEs = dict(
        Quality = dict(path = "%(subdet)s/%(prefix)sTimingClient/%(prefix)sTMT timing quality %(sm)s", otype = 'SM', btype = 'Crystal', kind = 'TH2F'),
        MeanSM = dict(path = "%(subdet)s/%(prefix)sTimingClient/%(prefix)sTMT timing mean %(sm)s", otype = 'SM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': -25., 'high': 25.}, yaxis = {'title': 'time (ns)'}),
        MeanAll = dict(path = "%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTMT%(suffix)s timing mean 1D summary", otype = 'Ecal3P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': -25., 'high': 25., 'title': 'time (ns)'}),
        FwdBkwdDiff = dict(path = "%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing %(prefix)s+ - %(prefix)s-", otype = 'Ecal2P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': -5., 'high': 5., 'title': 'time (ns)'}),
        FwdvBkwd = dict(path = "%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing %(prefix)s+ vs %(prefix)s-", otype = 'Ecal2P', btype = 'User', kind = 'TH2F', xaxis = {'nbins': 50, 'low': -25., 'high': 25.}, yaxis = {'nbins': 50, 'low': -25., 'high': 25., 'title': 'time (ns)'}),
        RMSMap = dict(path = "%(subdet)s/%(prefix)sTimingClient/%(prefix)sTMT timing rms %(sm)s", otype = 'SM', btype = 'Crystal', kind = 'TH2F', zaxis = {'title': 'rms (ns)'}),
        RMSAll = dict(path = "%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTMT%(suffix)s timing rms 1D summary", otype = 'Ecal3P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 10., 'title': 'time (ns)'}),
        ProjEta = dict(path = "%(subdet)s/%(prefix)sTimingClient/%(prefix)sTMT timing projection eta%(suffix)s", otype = 'Ecal3P', btype = 'ProjEta', kind = 'TProfile', yaxis = {'title': 'time (ns)'}),
        ProjPhi = dict(path = "%(subdet)s/%(prefix)sTimingClient/%(prefix)sTMT timing projection phi%(suffix)s", otype = 'Ecal3P', btype = 'ProjPhi', kind = 'TProfile', yaxis = {'title': 'time (ns)'}),
        QualitySummary = dict(path = "%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTMT%(suffix)s timing quality summary", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TH2F')
    ),
    sources = dict(
        TimeAllMap = ecalTimingTask['MEs']['TimeAllMap'],
        TimeMap = ecalTimingTask['MEs']['TimeMap']
    )
)
