from DQM.EcalBarrelMonitorTasks.TrigPrimTask_cfi import ecalTrigPrimTask

ecalTrigPrimClient = dict(
    MEs = dict(
        TimingSummary = dict(path = "%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTTT%(suffix)s Trigger Primitives Timing summary", otype = 'Ecal3P', btype = 'TriggerTower', kind = 'TH2F', zaxis = {'title': 'TP data matching emulator'}),
        NonSingleSummary = dict(path = "%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTTT%(suffix)s Trigger Primitives Non Single Timing summary", otype = 'Ecal3P', btype = 'TriggerTower', kind = 'TH2F'),
        EmulQualitySummary = dict(path = "%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTTT%(suffix)s emulator error quality summary", otype = 'Ecal3P', btype = 'TriggerTower', kind = 'TH2F')
    ),
    sources = dict(
        EtRealMap = ecalTrigPrimTask['MEs']['EtRealMap'],
        EtEmulError = ecalTrigPrimTask['MEs']['EtEmulError'],
        MatchedIndex = ecalTrigPrimTask['MEs']['MatchedIndex']
    )
)
