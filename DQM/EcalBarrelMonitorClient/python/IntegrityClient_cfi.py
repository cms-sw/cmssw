from DQM.EcalBarrelMonitorTasks.OccupancyTask_cfi import ecalOccupancyTask
from DQM.EcalBarrelMonitorTasks.IntegrityTask_cfi import ecalIntegrityTask

ecalIntegrityClient = dict(
    errFractionThreshold = 0.01,
    MEs = dict(
        Quality = dict(path = "%(subdet)s/%(prefix)sIntegrityClient/%(prefix)sIT data integrity quality %(sm)s", otype = 'SM', btype = 'Crystal', kind = 'TH2F'),
        QualitySummary = dict(path = "%(subdet)s/%(prefix)sSummaryClient/%(prefix)sIT%(suffix)s integrity quality summary", otype = 'Ecal3P', btype = 'Crystal', kind = 'TH2F')
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
