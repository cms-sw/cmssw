from DQM.EcalBarrelMonitorTasks.TestPulseTask_cfi import testPulseTask

testPulseClient = dict(
    amplitudeThresholdG1 = 100.,
    amplitudeThresholdG6 = 100.,
    amplitudeThresholdG12 = 100.,
    toleranceRMSG1 = 20.,
    toleranceRMSG6 = 20.,
    toleranceRMSG12 = 20.,
    PNAmplitudeThresholdG1 = 200. / 16.,
    PNAmplitudeThresholdG16 = 200.,
    tolerancePNRMSG1 = 20.,
    tolerancePNRMSG16 = 20.,
    MEs = dict(
        Quality = dict(path = 'TestPulse/Gain%(gain)s/Quality/TestPulseClient testpulse quality G%(gain)s', otype = 'SM', btype = 'Crystal', kind = 'TH2F'),
        AmplitudeMean = dict(path = 'TestPulse/Gain%(gain)s/AmplitudeMean/TestPulseClient amplitude mean G%(gain)s', otype = 'SM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 120, 'low': 170., 'high': 230.}),
        AmplitudeRMS = dict(path = "TestPulse/Gain%(gain)s/RMS/TestPulseClient rms G%(gain)s", otype = 'SM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 10.}),
        PNAmplitudeRMS = dict(path = 'PN/TestPulse/Gain%(pngain)s/RMS/TestPulseClient PN rms G%(pngain)s', otype = 'SMMEM', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 50.}),
        QualitySummary = dict(path = 'Summary/TestPulseClient testpulse quality G%(gain)s', otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F'),
        PNQualitySummary = dict(path = 'Summary/TestPulseClient PN quality G%(pngain)s', otype = 'MEM', btype = 'Crystal', kind = 'TH2F')
    ),
    sources = dict(
        Amplitude = testPulseTask['MEs']['Amplitude'],
        PNAmplitude = testPulseTask['MEs']['PNAmplitude']
    )
)
