from DQM.EcalBarrelMonitorTasks.PresampleTask_cfi import presampleTask

presampleClient = dict(
    minChannelEntries = 3,
    minTowerEntries = 30,
    expectedMean = 200.,
    meanThreshold = 25.,
    rmsThreshold = 3.,
    rmsThresholdHighEta = 6.,
    MEs = dict(
        Quality = dict(path = "Presample/Quality/PresampleClient presample quality", otype = 'SM', btype = 'Crystal', kind = 'TH2F'),
        Mean = dict(path = "Presample/Mean/PresampleClient mean", otype = 'SM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 120, 'low': 170., 'high': 230.}),
        MeanDCC = dict(path = "Presample/Mean/PresampleClient DCC mean", otype = 'Ecal2P', btype = 'DCC', kind = 'TProfile', yaxis = {'nbins': 120, 'low': 170., 'high': 230.}),
        RMS = dict(path = "Presample/RMS/PresampleClient rms", otype = 'SM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 10.}),
        RMSMap = dict(path = "Presample/RMSMap/PresampleClient rms", otype = 'SM', btype = 'Crystal', kind = 'TH2F'),
        QualitySummary = dict(path = "Summary/PresampleClient presample quality", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F')
    ),
    sources = dict(
        Pedestal = presampleTask['MEs']['Pedestal']
    )
)

