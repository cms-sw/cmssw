from DQM.EcalBarrelMonitorTasks.RawDataTask_cfi import rawDataTask

rawDataClient = dict(
    synchErrorThreshold = 10,
    MEs = dict(
        QualitySummary = dict(path = "Summary/RawDataClient FE status quality", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F')
    ),
    sources = dict(
        L1ADCC = rawDataTask['MEs']['L1ADCC'],
        FEStatus = rawDataTask['MEs']['FEStatus']
    )
)

