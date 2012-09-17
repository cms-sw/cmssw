from DQM.EcalBarrelMonitorTasks.PNDiodeTask_cfi import ecalPnDiodeTask

ecalPnIntegrityClient = dict(
    errFractionThreshold = 0.01,
    MEs = dict(
        QualitySummary = dict(path = "%(subdet)s/%(prefix)sSummaryClient/%(prefix)sIT PN integrity quality summary", otype = 'MEM2P', btype = 'Crystal', kind = 'TH2F')
    ),
    sources = dict(
        Occupancy = ecalPnDiodeTask['MEs']['Occupancy'],
        MEMChId = ecalPnDiodeTask['MEs']['MEMChId'],
        MEMGain = ecalPnDiodeTask['MEs']['MEMGain'],
        MEMBlockSize = ecalPnDiodeTask['MEs']['MEMBlockSize'],
        MEMTowerId = ecalPnDiodeTask['MEs']['MEMTowerId']
    )
)
