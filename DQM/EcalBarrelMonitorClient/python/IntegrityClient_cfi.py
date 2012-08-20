integrityClient = dict(
    errFractionThreshold = 0.01,
    sources = dict(
        Occupancy = ['OccupancyTask', 'Digi'],
        Gain = ['IntegrityTask', 'Gain'],
        ChId = ['IntegrityTask', 'ChId'],
        GainSwitch = ['IntegrityTask', 'GainSwitch'],
        TowerId = ['IntegrityTask', 'TowerId'],
        BlockSize = ['IntegrityTask', 'BlockSize']
    )
)

integrityClientPaths = dict(
    Quality = "Integrity/Quality/IntegrityClient data integrity quality",
    QualitySummary = "Summary/IntegrityClient data integrity quality"
)
