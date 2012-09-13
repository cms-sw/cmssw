from DQM.EcalBarrelMonitorTasks.LaserTask_cfi import ecalLaserTask

ecalLaserClient = dict(
    minChannelEntries = 3,
    expectedAmplitudeL1 = 1700.0,
    expectedAmplitudeL2 = 1300.0,
    expectedAmplitudeL3 = 1700.0,
    expectedAmplitudeL4 = 1700.0,
    toleranceAmplitudeL1 = 0.1, #relative to expected amplitude
    toleranceAmplitudeL2 = 0.1,
    toleranceAmplitudeL3 = 0.1,
    toleranceAmplitudeL4 = 0.1,
    toleranceAmpRMSRatioL1 = 0.3, #relative to mean amplitude
    toleranceAmpRMSRatioL2 = 0.3,
    toleranceAmpRMSRatioL3 = 0.3,
    toleranceAmpRMSRatioL4 = 0.3,
    expectedTimingL1 = 4.2,
    expectedTimingL2 = 4.2,
    expectedTimingL3 = 4.2,
    expectedTimingL4 = 4.2,
    toleranceTimingL1 = 0.5,
    toleranceTimingL2 = 0.5,
    toleranceTimingL3 = 0.5,
    toleranceTimingL4 = 0.5,
    toleranceTimRMSL1 = 0.4,
    toleranceTimRMSL2 = 0.4,
    toleranceTimRMSL3 = 0.4,
    toleranceTimRMSL4 = 0.4,
    expectedPNAmplitudeL1 = 800.0,
    expectedPNAmplitudeL2 = 800.0,
    expectedPNAmplitudeL3 = 800.0,
    expectedPNAmplitudeL4 = 800.0,
    tolerancePNAmpL1 = 0.1,
    tolerancePNAmpL2 = 0.1,
    tolerancePNAmpL3 = 0.1,
    tolerancePNAmpL4 = 0.1,
    tolerancePNRMSRatioL1 = 0.3,
    tolerancePNRMSRatioL2 = 0.3,
    tolerancePNRMSRatioL3 = 0.3,
    tolerancePNRMSRatioL4 = 0.3,
    forwardFactor = 0.5,
    MEs = dict(
        Quality = dict(path = 'Laser/Laser%(wl)s/Quality/LaserClient laser quality L%(wl)s', otype = 'SM', btype = 'Crystal', kind = 'TH2F', multi = 4),
        AmplitudeMean = dict(path = "Laser/Laser%(wl)s/Amplitude/Mean/LaserClient amplitude mean L%(wl)s", otype = 'SM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 4096.}, multi = 4),
        AmplitudeRMS = dict(path = "Laser/Laser%(wl)s/Amplitude/RMS/LaserClient amplitude RMS L%(wl)s", otype = 'SM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 800.}, multi = 4),
        TimingMean = dict(path = 'Laser/Laser%(wl)s/Timing/Mean/LaserClient timing mean L%(wl)s', otype = 'SM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 3.5, 'high': 5.5}, multi = 4),
        TimingRMS = dict(path = 'Laser/Laser%(wl)s/Timing/RMS/LaserClient timing RMS L%(wl)s', otype = 'SM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 0.5}, multi = 4),
        QualitySummary = dict(path = 'Summary/LaserClient laser quality L%(wl)s', otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F', multi = 4),
        PNQualitySummary = dict(path = 'Summary/LaserClient PN quality L%(wl)s', otype = 'MEM', btype = 'Crystal', kind = 'TH2F', multi = 4)
    ),
    sources = dict(
        Amplitude = ecalLaserTask['MEs']['Amplitude'],
        Timing = ecalLaserTask['MEs']['Timing'],
        PNAmplitude = ecalLaserTask['MEs']['PNAmplitude']
    )
)
