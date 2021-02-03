import FWCore.ParameterSet.Config as cms

ecalDigitizer = cms.PSet(
    ConstantTerm = cms.double(0.003),
    EBCorrNoiseMatrixG01 = cms.vdouble(
        1.0, 0.73354, 0.64442, 0.58851, 0.55425,
        0.53082, 0.51916, 0.51097, 0.50732, 0.50409
    ),
    EBCorrNoiseMatrixG06 = cms.vdouble(
        1.0, 0.70946, 0.58021, 0.49846, 0.45006,
        0.41366, 0.39699, 0.38478, 0.37847, 0.37055
    ),
    EBCorrNoiseMatrixG12 = cms.vdouble(
        1.0, 0.71073, 0.55721, 0.46089, 0.40449,
        0.35931, 0.33924, 0.32439, 0.31581, 0.30481
    ),
    EBdigiCollection = cms.string(''),
    EBs25notContainment = cms.double(0.9675),
    EECorrNoiseMatrixG01 = cms.vdouble(
        1.0, 0.72698, 0.62048, 0.55691, 0.51848,
        0.49147, 0.47813, 0.47007, 0.46621, 0.46265
    ),
    EECorrNoiseMatrixG06 = cms.vdouble(
        1.0, 0.71217, 0.47464, 0.34056, 0.26282,
        0.20287, 0.17734, 0.16256, 0.15618, 0.14443
    ),
    EECorrNoiseMatrixG12 = cms.vdouble(
        1.0, 0.71373, 0.44825, 0.30152, 0.21609,
        0.14786, 0.11772, 0.10165, 0.09465, 0.08098
    ),
    EEdigiCollection = cms.string(''),
    EEs25notContainment = cms.double(0.968),
    ESdigiCollection = cms.string(''),
    EcalPreMixStage1 = cms.bool(False),
    EcalPreMixStage2 = cms.bool(False),
    UseLCcorrection = cms.untracked.bool(True),
    accumulatorType = cms.string('EcalDigiProducer'),
    apdAddToBarrel = cms.bool(False),
    apdDigiTag = cms.string('APD'),
    apdDoPEStats = cms.bool(True),
    apdNonlParms = cms.vdouble(
        1.48, -3.75, 1.81, 1.26, 2.0,
        45, 1.0
    ),
    apdSeparateDigi = cms.bool(True),
    apdShapeTau = cms.double(40.5),
    apdShapeTstart = cms.double(74.5),
    apdSimToPEHigh = cms.double(88200000.0),
    apdSimToPELow = cms.double(2450000.0),
    apdTimeOffWidth = cms.double(0.8),
    apdTimeOffset = cms.double(-13.5),
    applyConstantTerm = cms.bool(True),
    binOfMaximum = cms.int32(6),
    cosmicsPhase = cms.bool(False),
    cosmicsShift = cms.double(0.0),
    doEB = cms.bool(True),
    doEE = cms.bool(False),
    doENoise = cms.bool(True),
    doES = cms.bool(False),
    doESNoise = cms.bool(True),
    doFast = cms.bool(True),
    doPhotostatistics = cms.bool(True),
    hitsProducer = cms.string('g4SimHits'),
    makeDigiSimLinks = cms.untracked.bool(False),
    photoelectronsToAnalogBarrel = cms.double(0.000444444),
    photoelectronsToAnalogEndcap = cms.double(0.000555555),
    readoutFrameSize = cms.int32(10),
    samplingFactor = cms.double(1.0),
    simHitToPhotoelectronsBarrel = cms.double(2250.0),
    simHitToPhotoelectronsEndcap = cms.double(1800.0),
    syncPhase = cms.bool(True),
    timeDependent = cms.bool(False),
    timePhase = cms.double(0.0)
)