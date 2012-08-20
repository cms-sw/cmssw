from DQM.EcalBarrelMonitorTasks.OccupancyTask_cfi import occupancyTaskPaths

occupancyClient = dict(
    minHits = 20,
    deviationThreshold = 100.0,
    sources = dict(
        Digi = ['OccupancyTask', 'Digi'],
        RecHitThr = ['OccupancyTask', 'RecHitThr'],
        TPDigiThr = ['OccupancyTask', 'TPDigiThr']
    )
)

occupancyClientPaths = dict(
    HotDigi = "Occupancy/HotCells/Digi/",
    HotRecHitThr = "Occupancy/HotCells/RecHitThres/",
    HotTPDigiThr = "Occupancy/HotCells/TPDigiThres/",
    QualitySummary = "Summary/OccupancyClient hot cell quality"
)
