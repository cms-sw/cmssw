from DQM.EcalBarrelMonitorTasks.TrigPrimTask_cfi import ecalTrigPrimTask

ecalTrigPrimClient = dict(
    MEs = dict(
        TimingSummary = dict(path = "TriggerPrimitives/TPClient TP timing", otype = 'Ecal2P', btype = 'TriggerTower', kind = 'TH2F'),
        NonSingleSummary = dict(path = "TriggerPrimitives/TPClient TP match non-single timing", otype = 'Ecal2P', btype = 'TriggerTower', kind = 'TH2F'),
        EmulQualitySummary = dict(path = "Summary/TPClient TP emulation quality", otype = 'Ecal2P', btype = 'TriggerTower', kind = 'TH2F')
    ),
    sources = dict(
        EtRealMap = ecalTrigPrimTask['MEs']['EtRealMap'],
        EtEmulError = ecalTrigPrimTask['MEs']['EtEmulError'],
        MatchedIndex = ecalTrigPrimTask['MEs']['MatchedIndex']
    )
)
