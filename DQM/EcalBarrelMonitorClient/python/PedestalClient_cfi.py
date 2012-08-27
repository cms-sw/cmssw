from DQM.EcalBarrelMonitorTasks.PedestalTask_cfi import pedestalTask

pedestalClient = dict(
    expectedMeanG1 = 200.,
    expectedMeanG6 = 200.,
    expectedMeanG12 = 200.,
    toleranceMeanG1 = 25.,
    toleranceMeanG6 = 25.,
    toleranceMeanG12 = 25.,
    toleranceRMSG1 = 1.,
    toleranceRMSG6 = 1.2,
    toleranceRMSG12 = 2.,
    expectedPNMeanG1 = 750.,
    expectedPNMeanG16 = 750.,
    tolerancePNMeanG1 = 100.,
    tolerancePNMeanG16 = 100.,
    tolerancePNRMSG1 = 20.,
    tolerancePNRMSG16 = 20.,
    MEs = dict(
        Quality = dict(path = 'Pedestal/Gain%(gain)s/Quality/PedestalClient pedestal quality G%(gain)s', otype = 'SM', btype = 'Crystal', kind = 'TH2F'),
        Mean = dict(path = 'Pedestal/Gain%(gain)s/Mean/PedestalClient mean G%(gain)s', otype = 'SM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 120, 'low': 170., 'high': 230.}),
        RMS = dict(path = "Pedestal/Gain%(gain)s/RMS/PedestalClient rms G%(gain)s", otype = 'SM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 10.}),
        PNRMS = dict(path = 'Pedestal/PN/Gain%(pngain)s/RMS/PedestalClient PN rms G%(pngain)s', otype = 'SMMEM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 50.}),
        QualitySummary = dict(path = 'Summary/PedestalClient pedestal quality G%(gain)s', otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F'),
        PNQualitySummary = dict(path = 'Summary/PedestalClient PN quality G%(pngain)s', otype = 'MEM', btype = 'Crystal', kind = 'TH2F')
    ),
    sources = dict(
        Pedestal = pedestalTask['MEs']['Pedestal'],
        PNPedestal = pedestalTask['MEs']['PNPedestal']
    )
)
