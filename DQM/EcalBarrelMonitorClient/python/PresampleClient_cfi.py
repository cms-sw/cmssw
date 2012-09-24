from DQM.EcalBarrelMonitorTasks.PresampleTask_cfi import ecalPresampleTask

ecalPresampleClient = dict(
    minChannelEntries = 6,
    minTowerEntries = 30,
    expectedMean = 200.,
    toleranceMean = 25.,
    toleranceRMS = 3.,
    toleranceRMSFwd = 6.,
    MEs = dict(
        Quality = dict(path = "%(subdet)s/%(prefix)sPedestalOnlineClient/%(prefix)sPOT pedestal quality G12 %(sm)s", otype = 'SM', btype = 'Crystal', kind = 'TH2F'),
        Mean = dict(path = "%(subdet)s/%(prefix)sPedestalOnlineClient/%(prefix)sPOT pedestal mean G12 %(sm)s", otype = 'SM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 120, 'low': 170., 'high': 230.}),
        RMS = dict(path = "%(subdet)s/%(prefix)sPedestalOnlineClient/%(prefix)sPOT pedestal rms G12 %(sm)s", otype = 'SM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 10.}),
        RMSMap = dict(path = "%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPOT%(suffix)s pedestal G12 RMS map", otype = 'Ecal3P', btype = 'Crystal', kind = 'TH2F', zaxis = {'title': 'RMS'}),
        QualitySummary = dict(path = "%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPOT%(suffix)s pedestal quality summary G12", otype = 'Ecal3P', btype = 'Crystal', kind = 'TH2F'),
        TrendMean = dict(path = 'Ecal/Trends/PresampleClient %(prefix)s pedestal mean max - min', otype = 'Ecal2P', btype = 'Trend', kind = 'TProfile'),
        TrendRMS = dict(path = 'Ecal/Trends/PresampleClient %(prefix)s pedestal rms max', otype = 'Ecal2P', btype = 'Trend', kind = 'TProfile')
    ),
    sources = dict(
        Pedestal = ecalPresampleTask['MEs']['Pedestal']
    )
)

