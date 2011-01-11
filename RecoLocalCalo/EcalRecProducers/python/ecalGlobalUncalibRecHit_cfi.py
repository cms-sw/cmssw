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
    EEtimeConstantTerm= cms.double(.6),
    EEtimeNconst      = cms.double(31.8),
    outOfTimeThresholdGain12pEB    = cms.double(5),      # times estimated precision
    outOfTimeThresholdGain12mEB    = cms.double(5),      # times estimated precision
    #outOfTimeThresholdGain61pEB    = cms.double(5),      # times estimated precision
    outOfTimeThresholdGain61pEB    = cms.double(1.e+05), # times estimated precision
    outOfTimeThresholdGain61mEB    = cms.double(5),      # times estimated precision
    outOfTimeThresholdGain12pEE    = cms.double(5),      # times estimated precision
    outOfTimeThresholdGain12mEE    = cms.double(5),      # times estimated precision
    #outOfTimeThresholdGain61pEE    = cms.double(5),      # times estimated precision
    outOfTimeThresholdGain61pEE    = cms.double(1.e+05), # times estimated precision
    outOfTimeThresholdGain61mEE    = cms.double(5),      # times estimated precision
    amplitudeThresholdEB    = cms.double(10),
    amplitudeThresholdEE    = cms.double(10),
    #amplitude-dependent time corrections; EE and EB have separate corrections
    #EXtimeCorrAmplitudes (ADC) and EXtimeCorrShifts (ns) need to have the same number of elements
    #Bins must be ordered in amplitude. First-last bins take care of under-overflows.
    doEBtimeCorrection = cms.bool(False),
    doEEtimeCorrection = cms.bool(False),

    EBtimeCorrAmplitudeBins = cms.vdouble(
    7.9,    8.9,    10.0,   11.2,   12.5,   14.1,   15.8,   17.7,    19.9,    22.3,   25.0,   28.1,   31.5,   35.3,   39.7,
    44.5,   49.9,   56.0,   62.8,   70.5,   79.1,   88.8,   99.6,    111.7,   125.4,  140.7,  157.9,  177.1,  198.7,  223.0,
    250.2,  280.7,  315.0,  353.4,  396.5,  444.9,  499.2,  560.1,   628.4,   705.1,  791.1,  887.7,  996.0,  1117.5, 1253.9,
    1406.8, 1578.5, 1771.1, 1987.2, 2229.7, 2501.8, 2807.0, 3149.5,  3533.8 , 3895.9, 3896.0 ),

    EBtimeCorrShiftBins     = cms.vdouble(
    -1.641,   -1.641,   -1.641,   -1.641,   -1.537,   -1.301,   -1.121,   -0.943,   -0.795,   -0.679,   -0.590,   -0.520,   -0.465,   -0.421,   -0.381,
    -0.345,   -0.318,   -0.293,   -0.273,   -0.253,   -0.235,   -0.220,   -0.206,   -0.196,   -0.188,   -0.181,   -0.167,   -0.148,   -0.136,   -0.130,
    -0.118,   -0.105,   -0.097,   -0.082,   -0.073,   -0.053,   -0.039,   -0.025,   -0.034,   -0.022,   -0.028,   -0.015,    0.025,    0.050,    0.069,
     0.107,    0.101,    0.130,    0.189,    0.191,    0.208,    0.237,    0.244,   0.251,     0.251,   -1.000 ),  

    EEtimeCorrAmplitudeBins = cms.vdouble(
    15.7,     17.6,     19.7,     22.1,     24.8,     27.9,     31.3,     35.1,     39.4,     44.2,     49.6,     55.6,     62.4,     70.0,     78.6,
    88.1,     98.9,     111.0,    124.5,    139.7,    156.7,    175.9,    197.3,    221.4,    248.4,    278.7,    312.7,    350.9,    393.7,    441.7,
    495.6,    556.1,    624.0,    700.1,    785.5,    881.4,    988.9,    1109.6,   1245.0,   1396.9,   1567.3,   1758.6,   1973.1,   2213.9,   2484.0,
    2787.1,   3127.2,   3508.8,   3936.9,   4417.3,   4956.3,   5561.1,   6239.6,   7001.0  ),

    EEtimeCorrShiftBins     = cms.vdouble(
    -1.052,   -1.052,   -1.052,   -1.052,   -0.719,   -0.531,   -0.383,   -0.271,   -0.190,   -0.145,   -0.107,   -0.077,   -0.076,   -0.072,   -0.075,
    -0.068,   -0.065,   -0.067,   -0.072,   -0.075,   -0.070,   -0.058,   -0.055,   -0.053,   -0.042,   -0.040,   -0.034,   -0.025,   -0.015,   -0.009,
    0.006,     0.013,    0.004,    0.003,   -0.013,    0.005,    0.029,    0.057,    0.076,    0.098,    0.123,    0.160,    0.195,    0.225,    0.253,
    0.278,     0.320,    0.360,    0.385,    0.358,    0.397,    0.367,    0.434,    0.428   ),                                     

    ebSpikeThreshold = cms.double(1.042),

    ebPulseShape = cms.vdouble( 5.2e-05,-5.26e-05 , 6.66e-05, 0.1168, 0.7575, 1.,  0.8876, 0.6732, 0.4741,  0.3194 ),
    eePulseShape = cms.vdouble( 5.2e-05,-5.26e-05 , 6.66e-05, 0.1168, 0.7575, 1.,  0.8876, 0.6732, 0.4741,  0.3194 ),

    kPoorRecoFlagEB = cms.bool(False),
    kPoorRecoFlagEE = cms.bool(False),
    chi2ThreshEB_ = cms.double(33.0),
    chi2ThreshEE_ = cms.double(33.0),
    EBchi2Parameters = cms.vdouble(2.122, 0.022, 2.122, 0.022),
    EEchi2Parameters = cms.vdouble(2.122, 0.022, 2.122, 0.022),
   
    algo = cms.string("EcalUncalibRecHitWorkerGlobal")
)
