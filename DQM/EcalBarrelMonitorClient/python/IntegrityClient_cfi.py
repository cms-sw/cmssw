from DQM.EcalBarrelMonitorTasks.OccupancyTask_cfi import ecalOccupancyTask
from DQM.EcalBarrelMonitorTasks.IntegrityTask_cfi import ecalIntegrityTask

ecalIntegrityClient = dict(
    errFractionThreshold = 0.01,
    MEs = dict(
        Quality = dict(path = "Integrity/Quality/IntegrityClient data integrity quality", otype = 'SM', btype = 'Crystal', kind = 'TH2F'),
        QualitySummary = dict(path = "Summary/IntegrityClient data integrity quality", otype = 'Ecal2P', btype = 'Crystal', kind = 'TH2F')
    ),
    sources = dict(
        Occupancy = ecalOccupancyTask['MEs']['Digi'],            
        Gain = ecalIntegrityTask['MEs']['Gain'],
        ChId = ecalIntegrityTask['MEs']['ChId'],
        GainSwitch = ecalIntegrityTask['MEs']['GainSwitch'],
        TowerId = ecalIntegrityTask['MEs']['TowerId'],
        BlockSize = ecalIntegrityTask['MEs']['BlockSize']
    )
)
