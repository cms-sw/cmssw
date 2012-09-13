from DQM.EcalBarrelMonitorTasks.TestPulseTask_cfi import ecalTestPulseTask

ecalTestPulseClient = dict(
    amplitudeThresholdG1 = 100.,
    amplitudeThresholdG6 = 100.,
    amplitudeThresholdG12 = 100.,
    toleranceRMSG1 = 10.,
    toleranceRMSG6 = 10.,
    toleranceRMSG12 = 10.,
    PNAmplitudeThresholdG1 = 200. / 16.,
    PNAmplitudeThresholdG16 = 200.,
    tolerancePNRMSG1 = 20.,
    tolerancePNRMSG16 = 20.,
    MEs = dict(
        Quality = dict(path = 'TestPulse/Gain%(gain)s/Quality/TestPulseClient testpulse quality G%(gain)s', otype = 'SM', btype = 'Crystal', kind = 'TH2F', multi = 3),
        AmplitudeRMS = dict(path = "TestPulse/Gain%(gain)s/RMS/TestPulseClient rms G%(gain)s", otype = 'SM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 10.}, multi = 3),
        PNAmplitudeRMS = dict(path = 'PN/TestPulse/Gain%(pngain)s/RMS/TestPulseClient PN rms G%(pngain)s', otype = 'SMMEM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 50.}, multi = 2),
        QualitySummary = dict(path = 'Summary/TestPulseClient testpulse quality G%(gain)s', otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F', multi = 3),
        PNQualitySummary = dict(path = 'Summary/TestPulseClient PN quality G%(pngain)s', otype = 'MEM', btype = 'Crystal', kind = 'TH2F', multi = 2)
    ),
    sources = dict(
        Amplitude = ecalTestPulseTask['MEs']['Amplitude'],
        PNAmplitude = ecalTestPulseTask['MEs']['PNAmplitude']
    )
)
