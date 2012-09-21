from DQM.EcalBarrelMonitorTasks.TestPulseTask_cfi import ecalTestPulseTask

ecalTestPulseClient = dict(
    amplitudeThresholdG01 = 100.,
    amplitudeThresholdG06 = 100.,
    amplitudeThresholdG12 = 100.,
    toleranceRMSG01 = 10.,
    toleranceRMSG06 = 10.,
    toleranceRMSG12 = 10.,
    PNAmplitudeThresholdG01 = 200. / 16.,
    PNAmplitudeThresholdG16 = 200.,
    tolerancePNRMSG01 = 20.,
    tolerancePNRMSG16 = 20.,
    MEs = dict(
        Quality = dict(path = '%(subdet)s/%(prefix)sTestPulseClient/%(prefix)sTPT test pulse quality G%(gain)s %(sm)s', otype = 'SM', btype = 'Crystal', kind = 'TH2F', multi = 3),
        AmplitudeRMS = dict(path = "%(subdet)s/%(prefix)sTestPulseClient/%(prefix)sTPT test pulse rms G%(gain)s", otype = 'Ecal2P', btype = 'Crystal', kind = 'TH2F', zaxis = {'title': 'rms (ADC counts)'}, multi = 3),
        QualitySummary = dict(path = '%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTPT%(suffix)s test pulse quality G%(gain)s summary', otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TH2F', multi = 3),
        PNQualitySummary = dict(path = '%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTPT PN test pulse quality G%(pngain)s summary', otype = 'MEM2P', btype = 'Crystal', kind = 'TH2F', multi = 2)
    ),
    sources = dict(
        Amplitude = ecalTestPulseTask['MEs']['Amplitude'],
        PNAmplitude = ecalTestPulseTask['MEs']['PNAmplitude']
    )
)
