from DQM.EcalBarrelMonitorTasks.LedTask_cfi import ledTask

ledClient = dict(
    minChannelEntries = 3,
    expectedAmplitudeL1 = 1500.0,
    expectedAmplitudeL2 = 1500.0,
    amplitudeThresholdL1 = 1000.0,
    amplitudeThresholdL2 = 1000.0,
    amplitudeRMSThresholdL1 = 50.0,
    amplitudeRMSThresholdL2 = 50.0,
    expectedTimingL1 = 5.5,
    expectedTimingL2 = 5.5,
    timingThresholdL1 = 0.5,
    timingThresholdL2 = 0.5,
    timingRMSThresholdL1 = 0.2,
    timingRMSThresholdL2 = 0.2,
    expectedPNAmplitudeL1 = 800.0,
    expectedPNAmplitudeL2 = 800.0,
    pnAmplitudeThresholdL1 = 500.0,
    pnAmplitudeThresholdL2 = 500.0,
    pnAmplitudeRMSThresholdL1 = 100.0,
    pnAmplitudeRMSThresholdL2 = 100.0,
    towerThreshold = 0.1,
    MEs = dict(
        AmplitudeMean = dict(path = "Led/Led%(wl)s/Amplitude/Mean/LedClient amplitude mean L%(wl)s", otype = 'EESM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 4096.}),
        AmplitudeRMS = dict(path = "Led/Led%(wl)s/Amplitude/RMS/LedClient amplitude RMS L%(wl)s", otype = 'EESM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 400.}),
        TimingMean = dict(path = 'Led/Led%(wl)s/Timing/Mean/LedClient timing mean L%(wl)s', otype = 'EESM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 3.5, 'high': 5.5}),
        TimingRMS = dict(path = 'Led/Led%(wl)s/Timing/RMS/LedClient timing RMS L%(wl)s', otype = 'EESM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 0.5}),
        Quality = dict(path = 'Led/Led%(wl)s/Quality/LedClient led quality L%(wl)s', otype = 'EESM', btype = 'Crystal', kind = 'TH2F'),
        QualitySummary = dict(path = 'Summary/LedClient led quality L%(wl)s', otype = 'EE', btype = 'SuperCrystal', kind = 'TH2F'),
        PNQualitySummary = dict(path = 'Summary/LedClient PN quality L%(wl)s', otype = 'EEMEM', btype = 'Crystal', kind = 'TH2F')
    ),
    sources = dict(
        Amplitude = ledTask['MEs']['Amplitude'],
        Timing = ledTask['MEs']['Timing'],
        PNAmplitude = ledTask['MEs']['PNAmplitude']
    )
)
