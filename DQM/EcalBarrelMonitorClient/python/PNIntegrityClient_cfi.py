from DQM.EcalBarrelMonitorTasks.PNOccupancyTask_cfi import pnOccupancyTask
from DQM.EcalBarrelMonitorTasks.PNIntegrityTask_cfi import pnIntegrityTask

pnIntegrityClient = dict(
    errFractionThreshold = 0.01,
    MEs = dict(
        QualitySummary = "Integrity/PNQuality/PNIntegrityClient data integrity quality"
    ),
    sources = dict(
        Occupancy = pnOccupancyTask['MEs']['Digi'],
        MEMChId = pnIntegrityTask['MEs']['MEMChId'],
        MEMGain = pnIntegrityTask['MEs']['MEMGain'],
        MEMBlockSize = pnIntegrityTask['MEs']['MEMBlockSize'],
        MEMTowerId = pnIntegrityTask['MEs']['MEMTowerId']
    )
)
