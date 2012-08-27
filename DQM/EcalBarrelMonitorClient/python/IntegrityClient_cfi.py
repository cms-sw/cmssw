from DQM.EcalBarrelMonitorTasks.OccupancyTask_cfi import occupancyTask
from DQM.EcalBarrelMonitorTasks.IntegrityTask_cfi import integrityTask

integrityClient = dict(
    errFractionThreshold = 0.01,
    MEs = dict(
        Quality = dict(path = "Integrity/Quality/IntegrityClient data integrity quality", otype = 'SM', btype = 'Crystal', kind = 'TH2F'),
        QualitySummary = dict(path = "Summary/IntegrityClient data integrity quality", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F')
    ),
    sources = dict(
        Occupancy = occupancyTask['MEs']['Digi'],            
        Gain = integrityTask['MEs']['Gain'],
        ChId = integrityTask['MEs']['ChId'],
        GainSwitch = integrityTask['MEs']['GainSwitch'],
        TowerId = integrityTask['MEs']['TowerId'],
        BlockSize = integrityTask['MEs']['BlockSize']
    )
)
