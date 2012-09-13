from DQM.EcalBarrelMonitorTasks.OccupancyTask_cfi import ecalOccupancyTask

ecalOccupancyClient = dict(
    minHits = 20,
    deviationThreshold = 100.0,
    MEs = dict(
        HotDigi = dict(path = "Occupancy/HotCells/Digi/", otype = 'Channel', btype = 'Crystal', kind = 'TH1F'),
        HotRecHitThr = dict(path = "Occupancy/HotCells/RecHitThres/", otype = 'Channel', btype = 'Crystal', kind = 'TH1F'),
        HotTPDigiThr = dict(path = "Occupancy/HotCells/TPDigiThres/", otype = 'Channel', btype = 'TriggerTower', kind = 'TH1F'),
        QualitySummary = dict(path = "Summary/OccupancyClient hot cell quality", otype = 'Ecal2P', btype = 'Crystal', kind = 'TH2F')
    ),
    sources = dict(
        Digi = ecalOccupancyTask['MEs']['Digi'],
        RecHitThr = ecalOccupancyTask['MEs']['RecHitThr'],
        TPDigiThr = ecalOccupancyTask['MEs']['TPDigiThr']
    )
)

