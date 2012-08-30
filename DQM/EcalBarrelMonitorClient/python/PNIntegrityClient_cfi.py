from DQM.EcalBarrelMonitorTasks.PNDiodeTask_cfi import pnDiodeTask

pnIntegrityClient = dict(
    errFractionThreshold = 0.01,
    MEs = dict(
        QualitySummary = dict(path = "Summary/PNIntegrityClient data integrity quality", otype = 'MEM', btype = 'Crystal', kind = 'TH2F')
    ),
    sources = dict(
        Occupancy = pnDiodeTask['MEs']['Occupancy'],
        MEMChId = pnDiodeTask['MEs']['MEMChId'],
        MEMGain = pnDiodeTask['MEs']['MEMGain'],
        MEMBlockSize = pnDiodeTask['MEs']['MEMBlockSize'],
        MEMTowerId = pnDiodeTask['MEs']['MEMTowerId']
    )
)
