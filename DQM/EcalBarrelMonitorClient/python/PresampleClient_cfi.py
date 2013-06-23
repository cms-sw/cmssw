presampleClient = dict(
    minChannelEntries = 3,
    minTowerEntries = 30,
    expectedMean = 200.,
    meanThreshold = 25.,
    rmsThreshold = 3.,
    rmsThresholdHighEta = 6.,
    noisyFracThreshold = 0.1
)

presampleClientPaths = dict(
    Quality = "Presample/Quality/PresampleClient presample quality",
    Mean = "Presample/Mean/PresampleClient mean",
    MeanDCC = "Presample/Mean/PresampleClient DCC mean",
    RMS = "Presample/RMS/PresampleClient rms",
    RMSMap = "Presample/RMSMap/PresampleClient rms",
    QualitySummary = "Summary/PresampleClient presample quality"
)
