from DQM.EcalBarrelMonitorTasks.RawDataTask_cfi import ecalRawDataTask

ecalRawDataClient = dict(
    synchErrorThreshold = 10,
    MEs = dict(
        QualitySummary = dict(path = "%(subdet)s/%(prefix)sSummaryClient/%(prefix)sSFT%(suffix)s front-end status summary", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TH2F')
    ),
    sources = dict(
        L1ADCC = ecalRawDataTask['MEs']['L1ADCC'],
        FEStatus = ecalRawDataTask['MEs']['FEStatus']
    )
)

