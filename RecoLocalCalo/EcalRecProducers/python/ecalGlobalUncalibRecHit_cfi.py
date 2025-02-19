import FWCore.ParameterSet.Config as cms

ecalGlobalUncalibRecHit = cms.EDProducer("EcalUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    EEdigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    EBhitCollection = cms.string("EcalUncalibRecHitsEB"),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),

    # for ratio method
    EBtimeFitParameters = cms.vdouble(-2.015452e+00, 3.130702e+00, -1.234730e+01, 4.188921e+01, -8.283944e+01, 9.101147e+01, -5.035761e+01, 1.105621e+01),
    EEtimeFitParameters = cms.vdouble(-2.390548e+00, 3.553628e+00, -1.762341e+01, 6.767538e+01, -1.332130e+02, 1.407432e+02, -7.541106e+01, 1.620277e+01),
    EBamplitudeFitParameters = cms.vdouble(1.138,1.652),
    EEamplitudeFitParameters = cms.vdouble(1.890,1.400),
    EBtimeFitLimits_Lower = cms.double(0.2),
    EBtimeFitLimits_Upper = cms.double(1.4),
    EEtimeFitLimits_Lower = cms.double(0.2),
    EEtimeFitLimits_Upper = cms.double(1.4),
    # for kOutOfTime flag
    EBtimeConstantTerm= cms.double(.6),
    EBtimeNconst      = cms.double(28.5),
    EEtimeConstantTerm= cms.double(1.0),
    EEtimeNconst      = cms.double(31.8),
    outOfTimeThresholdGain12pEB    = cms.double(5),      # times estimated precision
    outOfTimeThresholdGain12mEB    = cms.double(5),      # times estimated precision
    outOfTimeThresholdGain61pEB    = cms.double(5),      # times estimated precision
    outOfTimeThresholdGain61mEB    = cms.double(5),      # times estimated precision
    outOfTimeThresholdGain12pEE    = cms.double(10),      # times estimated precision
    outOfTimeThresholdGain12mEE    = cms.double(10),      # times estimated precision
    outOfTimeThresholdGain61pEE    = cms.double(10),      # times estimated precision
    outOfTimeThresholdGain61mEE    = cms.double(10),      # times estimated precision
    amplitudeThresholdEB    = cms.double(10),
    amplitudeThresholdEE    = cms.double(10),
    #amplitude-dependent time corrections; EE and EB have separate corrections
    #EXtimeCorrAmplitudes (ADC) and EXtimeCorrShifts (ns) need to have the same number of elements
    #Bins must be ordered in amplitude. First-last bins take care of under-overflows.
    doEBtimeCorrection = cms.bool(False),
    doEEtimeCorrection = cms.bool(False),

    EBtimeCorrAmplitudeBins = cms.vdouble(
    7.9,    8.9,    10,     11.2,   12.5,   14.1,   15.8,   17.7,   19.9,   22.3,   25,     28.1,   31.5,   35.3,   39.7,
    44.5,   49.9,   56,     62.8,   70.5,   79.1,   88.8,   99.6,   111.7,  125.4,  140.7,  157.9,  177.1,  198.7,  223,
    250.2,  280.7,  315,    353.4,  396.5,  444.9,  499.2,  560.1,  628.4,  705.1,  791.1,  887.7,  996,    1117.5, 1253.9,
    1406.8, 1578.5, 1771.1, 1987.2, 2229.7, 2501.8, 2807,   3149.5, 3533.8, 3895.9, 3896,   4311.8, 4837.9, 5428.2, 6090.6,
    6833.7, 7667.5, 8603.1, 9652.9, 10830,  12152,  13635,  15298,  17165,  19260,  21610),
                                         
    EBtimeCorrShiftBins     = cms.vdouble(
    -1.770, -1.770, -1.770, -1.770, -1.666, -1.430, -1.233, -1.012, -0.866, -0.736, -0.640, -0.561, -0.505, -0.452, -0.405,
    -0.363, -0.335, -0.305, -0.279, -0.260, -0.239, -0.220, -0.204, -0.191, -0.186, -0.177, -0.158, -0.137, -0.126, -0.115,
    -0.104, -0.096, -0.085, -0.064, -0.056, -0.036, -0.020, -0.006, -0.020, -0.009, -0.020, 0.005,  0.053,  0.076,  0.093,
    0.137,  0.143,  0.171,  0.222,  0.229,  0.271,  0.298,  0.312,  0.307,  0.254 , -0.997 ,-0.859 , -0.819, -0.775, -0.589,
    -0.428, -0.288, -0.434, -0.277, -0.210, -0.179, -0.134, 0.362,  0.152,  -0.282,  -0.382),
                                         
    EEtimeCorrAmplitudeBins = cms.vdouble(
    15.7,   17.6,   19.7,   22.1,   24.8,   27.9,   31.3,   35.1,   39.4,   44.2,   49.6,   55.6,   62.4,   70,     78.6,
    88.1,   98.9,   111,    124.5,  139.7,  156.7,  175.9,  197.3,  221.4,  248.4,  278.7,  312.7,  350.9,  393.7,  441.7,
    495.6,  556.1,  624,    700.1,  785.5,  881.4,  988.9,  1109.6, 1245,   1396.9, 1567.3, 1758.6, 1973.1, 2213.9, 2484,
    2787.1, 3127.2, 3508.8, 3936.9, 4417.3, 4956.3, 5561.1, 6239.6, 7001,   7522.8, 8440.7, 9470.6, 10626),
                                         
    EEtimeCorrShiftBins     = cms.vdouble(
    -0.896, -0.896, -0.896, -0.896, -0.563, -0.392, -0.287, -0.203, -0.135, -0.100, -0.068, -0.050, -0.060, -0.052, -0.055,
    -0.050, -0.052, -0.056, -0.055, -0.056, -0.048, -0.037, -0.038, -0.037, -0.025, -0.026, -0.024, -0.013, -0.003, 0.005,
    0.020,  0.026,  0.008,  0.007,  -0.006, 0.024,  0.045,  0.062,  0.085,  0.088 , 0.111 , 0.139,  0.156,  0.176,  0.210,
    0.242,  0.267,  0.301,  0.318,  0.278,  0.287,  0.218,  0.305,  0.245,  0.184,  -0.159, -0.095, 0.037),

    ebSpikeThreshold = cms.double(1.042),

    ebPulseShape = cms.vdouble( 5.2e-05,-5.26e-05 , 6.66e-05, 0.1168, 0.7575, 1.,  0.8876, 0.6732, 0.4741,  0.3194 ),
    eePulseShape = cms.vdouble( 5.2e-05,-5.26e-05 , 6.66e-05, 0.1168, 0.7575, 1.,  0.8876, 0.6732, 0.4741,  0.3194 ),

    kPoorRecoFlagEB = cms.bool(True),
    kPoorRecoFlagEE = cms.bool(False),
    chi2ThreshEB_ = cms.double(36.0),
    chi2ThreshEE_ = cms.double(95.0),
    EBchi2Parameters = cms.vdouble(2.122, 0.022, 2.122, 0.022),
    EEchi2Parameters = cms.vdouble(2.122, 0.022, 2.122, 0.022),
   
    algo = cms.string("EcalUncalibRecHitWorkerGlobal")
)
