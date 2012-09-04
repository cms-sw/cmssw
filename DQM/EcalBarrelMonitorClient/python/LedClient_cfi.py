from DQM.EcalBarrelMonitorTasks.LedTask_cfi import ledTask

ledClient = dict(
    minChannelEntries = 3,
    expectedAmplitudeL1 = 1700.0,
    expectedAmplitudeL2 = 1700.0,
    toleranceAmplitudeL1 = 1000.0,
    toleranceAmplitudeL2 = 1000.0,
    toleranceAmpRMSL1 = 400.,
    toleranceAmpRMSL2 = 400.,
    expectedTimingL1 = 4.2,
    expectedTimingL2 = 4.2,
    toleranceTimingL1 = 0.5,
    toleranceTimingL2 = 0.5,
    toleranceTimRMSL1 = 0.4,
    toleranceTimRMSL2 = 0.4,
    expectedPNAmplitudeL1 = 800.0,
    expectedPNAmplitudeL2 = 800.0,
    tolerancePNAmpL1 = 500.0,
    tolerancePNAmpL2 = 500.0,
    tolerancePNRMSL1 = 100.0,
    tolerancePNRMSL2 = 100.0,
    forwardFactor = 0.5,
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
