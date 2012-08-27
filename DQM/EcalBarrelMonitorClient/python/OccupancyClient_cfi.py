from DQM.EcalBarrelMonitorTasks.OccupancyTask_cfi import occupancyTask

occupancyClient = dict(
    minHits = 20,
    deviationThreshold = 100.0,
    MEs = dict(
        HotDigi = dict(path = "Occupancy/HotCells/Digi/", otype = 'Channel', btype = 'Crystal', kind = 'TH1F'),
        HotRecHitThr = dict(path = "Occupancy/HotCells/RecHitThres/", otype = 'Channel', btype = 'Crystal', kind = 'TH1F'),
        HotTPDigiThr = dict(path = "Occupancy/HotCells/TPDigiThres/", otype = 'Channel', btype = 'TriggerTower', kind = 'TH1F'),
        QualitySummary = dict(path = "Summary/OccupancyClient hot cell quality", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F')
    ),
    sources = dict(
        Digi = occupancyTask['MEs']['Digi'],
        RecHitThr = occupancyTask['MEs']['RecHitThr'],
        TPDigiThr = occupancyTask['MEs']['TPDigiThr']
    )
)

