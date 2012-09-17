from DQM.EcalBarrelMonitorTasks.OccupancyTask_cfi import ecalOccupancyTask

ecalOccupancyClient = dict(
    minHits = 20,
    deviationThreshold = 100.0,
    MEs = dict(
        HotDigi = dict(path = "Ecal/Errors/HotCells/Digi/", otype = 'Channel', btype = 'Crystal', kind = 'TH1F'),
        HotRecHitThr = dict(path = "Ecal/Errors/HotCells/RecHitThres/", otype = 'Channel', btype = 'Crystal', kind = 'TH1F'),
        HotTPDigiThr = dict(path = "Ecal/Errors/HotCells/TPDigiThres/", otype = 'Channel', btype = 'TriggerTower', kind = 'TH1F'),
        QualitySummary = dict(path = "%(subdet)s/%(prefix)sSummaryClient/%(prefix)sOT%(suffix)s hot cell quality summary", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TH2F')
    ),
    sources = dict(
        Digi = ecalOccupancyTask['MEs']['DigiAll'],
        RecHitThr = ecalOccupancyTask['MEs']['RecHitThrAll'],
        TPDigiThr = ecalOccupancyTask['MEs']['TPDigiThrAll']
    )
)

