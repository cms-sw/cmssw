from DQM.EcalBarrelMonitorTasks.LedTask_cfi import ecalLedTask

ecalLedClient = dict(
    minChannelEntries = 3,
    expectedAmplitudeL1 = 200.0,
    expectedAmplitudeL2 = 200.0,
    toleranceAmplitudeL1 = 0.1, # relative to expected amplitude
    toleranceAmplitudeL2 = 0.1,
    toleranceAmpRMSRatioL1 = 0.3, # relative to mean amplitude
    toleranceAmpRMSRatioL2 = 0.3,
    expectedTimingL1 = 4.2,
    expectedTimingL2 = 4.2,
    toleranceTimingL1 = 0.5,
    toleranceTimingL2 = 0.5,
    toleranceTimRMSL1 = 0.4,
    toleranceTimRMSL2 = 0.4,
    expectedPNAmplitudeL1 = 800.0,
    expectedPNAmplitudeL2 = 800.0,
    tolerancePNAmpL1 = 0.1,
    tolerancePNAmpL2 = 0.1,
    tolerancePNRMSRatioL1 = 0.3,
    tolerancePNRMSRatioL2 = 0.3,
    forwardFactor = 0.5,
    MEs = dict(
        AmplitudeMean = dict(path = "Led/Led%(wl)s/Amplitude/Mean/LedClient amplitude mean L%(wl)s", otype = 'EESM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 400.}, multi = 2),
        AmplitudeRMS = dict(path = "Led/Led%(wl)s/Amplitude/RMS/LedClient amplitude RMS L%(wl)s", otype = 'EESM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 400.}, multi = 2),
        TimingMean = dict(path = 'Led/Led%(wl)s/Timing/Mean/LedClient timing mean L%(wl)s', otype = 'EESM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 3.5, 'high': 5.5}, multi = 2),
        TimingRMS = dict(path = 'Led/Led%(wl)s/Timing/RMS/LedClient timing RMS L%(wl)s', otype = 'EESM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 0.5}, multi = 2),
        Quality = dict(path = 'Led/Led%(wl)s/Quality/LedClient led quality L%(wl)s', otype = 'EESM', btype = 'Crystal', kind = 'TH2F', multi = 2),
        QualitySummary = dict(path = 'Summary/LedClient led quality L%(wl)s', otype = 'EE', btype = 'SuperCrystal', kind = 'TH2F', multi = 2),
        PNQualitySummary = dict(path = 'Summary/LedClient PN quality L%(wl)s', otype = 'EEMEM', btype = 'Crystal', kind = 'TH2F', multi = 2)
    ),
    sources = dict(
        Amplitude = ecalLedTask['MEs']['Amplitude'],
        Timing = ecalLedTask['MEs']['Timing'],
        PNAmplitude = ecalLedTask['MEs']['PNAmplitude']
    )
)
