import FWCore.ParameterSet.Config as cms

hcalSimBlock = cms.PSet(
    DelivLuminosity = cms.double(0),
    HBDarkening = cms.bool(False),
    HEDarkening = cms.bool(False),
    HFDarkening = cms.bool(False),
    HFRecalParameterBlock = cms.PSet(
        HFdepthOneParameterA = cms.vdouble(
            0.004123, 0.00602, 0.008201, 0.010489, 0.013379,
            0.016997, 0.021464, 0.027371, 0.034195, 0.044807,
            0.058939, 0.125497
        ),
        HFdepthOneParameterB = cms.vdouble(
            -4e-06, -2e-06, 0.0, 4e-06, 1.5e-05,
            2.6e-05, 6.3e-05, 8.4e-05, 0.00016, 0.000107,
            0.000425, 0.000209
        ),
        HFdepthTwoParameterA = cms.vdouble(
            0.002861, 0.004168, 0.0064, 0.008388, 0.011601,
            0.014425, 0.018633, 0.023232, 0.028274, 0.035447,
            0.051579, 0.086593
        ),
        HFdepthTwoParameterB = cms.vdouble(
            -2e-06, -0.0, -7e-06, -6e-06, -2e-06,
            1e-06, 1.9e-05, 3.1e-05, 6.7e-05, 1.2e-05,
            0.000157, -3e-06
        )
    ),
    HcalPreMixStage1 = cms.bool(False),
    HcalPreMixStage2 = cms.bool(False),
    TestNumbering = cms.bool(True),
    debugCaloSamples = cms.bool(False),
    doEmpty = cms.bool(True),
    doHFWindow = cms.bool(False),
    doIonFeedback = cms.bool(True),
    doNeutralDensityFilter = cms.bool(True),
    doNoise = cms.bool(True),
    doThermalNoise = cms.bool(True),
    doTimeSlew = cms.bool(True),
    hb = cms.PSet(
        binOfMaximum = cms.int32(6),
        doPhotoStatistics = cms.bool(True),
        doSiPMSmearing = cms.bool(True),
        firstRing = cms.int32(1),
        readoutFrameSize = cms.int32(10),
        samplingFactors = cms.vdouble(
            125.44, 125.54, 125.32, 125.13, 124.46,
            125.01, 125.22, 125.48, 124.45, 125.9,
            125.83, 127.01, 126.82, 129.73, 131.83,
            143.52
        ),
        simHitToPhotoelectrons = cms.double(2000.0),
        sipmTau = cms.double(10.0),
        syncPhase = cms.bool(True),
        threshold_currentTDC = cms.double(18.7),
        timePhase = cms.double(6.0),
        timeSmearing = cms.bool(True)
    ),
    he = cms.PSet(
        binOfMaximum = cms.int32(6),
        doPhotoStatistics = cms.bool(True),
        doSiPMSmearing = cms.bool(True),
        firstRing = cms.int32(16),
        readoutFrameSize = cms.int32(10),
        samplingFactors = cms.vdouble(
            210.55, 197.93, 186.12, 189.64, 189.63,
            189.96, 190.03, 190.11, 190.18, 190.25,
            190.32, 190.4, 190.47, 190.54, 190.61,
            190.69, 190.83, 190.94, 190.94, 190.94,
            190.94, 190.94, 190.94, 190.94, 190.94,
            190.94, 190.94, 190.94, 190.94, 190.94,
            190.94, 190.94, 190.94, 190.94, 190.94,
            190.94, 190.94, 190.94, 190.94, 190.94,
            190.94, 190.94, 190.94, 190.94, 190.94,
            190.94, 190.94, 190.94, 190.94, 190.94,
            190.94, 190.94, 190.94, 190.94, 190.94,
            190.94, 190.94, 190.94, 190.94, 190.94,
            190.94, 190.94, 190.94, 190.94, 190.94,
            190.94, 190.94, 190.94, 190.94, 190.94,
            190.94, 190.94, 190.94, 190.94, 190.94,
            190.94, 190.94, 190.94, 190.94, 190.94,
            190.94, 190.94, 190.94, 190.94, 190.94
        ),
        simHitToPhotoelectrons = cms.double(2000.0),
        sipmTau = cms.double(10.0),
        syncPhase = cms.bool(True),
        threshold_currentTDC = cms.double(18.7),
        timePhase = cms.double(6.0),
        timeSmearing = cms.bool(True)
    ),
    hf1 = cms.PSet(
        binOfMaximum = cms.int32(2),
        doPhotoStatistics = cms.bool(True),
        doSiPMSmearing = cms.bool(False),
        photoelectronsToAnalog = cms.double(2.79),
        readoutFrameSize = cms.int32(3),
        samplingFactor = cms.double(0.335),
        simHitToPhotoelectrons = cms.double(6.0),
        sipmTau = cms.double(0.0),
        syncPhase = cms.bool(True),
        threshold_currentTDC = cms.double(-999.0),
        timePhase = cms.double(14.0)
    ),
    hf2 = cms.PSet(
        binOfMaximum = cms.int32(2),
        doPhotoStatistics = cms.bool(True),
        doSiPMSmearing = cms.bool(False),
        photoelectronsToAnalog = cms.double(1.843),
        readoutFrameSize = cms.int32(3),
        samplingFactor = cms.double(0.335),
        simHitToPhotoelectrons = cms.double(6.0),
        sipmTau = cms.double(0.0),
        syncPhase = cms.bool(True),
        threshold_currentTDC = cms.double(-999.0),
        timePhase = cms.double(13.0)
    ),
    hitsProducer = cms.string('g4SimHits'),
    ho = cms.PSet(
        binOfMaximum = cms.int32(5),
        doPhotoStatistics = cms.bool(True),
        doSiPMSmearing = cms.bool(False),
        firstRing = cms.int32(1),
        readoutFrameSize = cms.int32(10),
        samplingFactors = cms.vdouble(
            231.0, 231.0, 231.0, 231.0, 360.0,
            360.0, 360.0, 360.0, 360.0, 360.0,
            360.0, 360.0, 360.0, 360.0, 360.0
        ),
        siPMCode = cms.int32(1),
        simHitToPhotoelectrons = cms.double(4000.0),
        sipmTau = cms.double(5.0),
        syncPhase = cms.bool(True),
        threshold_currentTDC = cms.double(-999.0),
        timePhase = cms.double(5.0),
        timeSmearing = cms.bool(False)
    ),
    hoHamamatsu = cms.PSet(
        binOfMaximum = cms.int32(5),
        doPhotoStatistics = cms.bool(True),
        doSiPMSmearing = cms.bool(False),
        firstRing = cms.int32(1),
        readoutFrameSize = cms.int32(10),
        samplingFactors = cms.vdouble(
            231.0, 231.0, 231.0, 231.0, 360.0,
            360.0, 360.0, 360.0, 360.0, 360.0,
            360.0, 360.0, 360.0, 360.0, 360.0
        ),
        siPMCode = cms.int32(2),
        simHitToPhotoelectrons = cms.double(4000.0),
        sipmTau = cms.double(5.0),
        syncPhase = cms.bool(True),
        threshold_currentTDC = cms.double(-999.0),
        timePhase = cms.double(5.0),
        timeSmearing = cms.bool(False)
    ),
    hoZecotek = cms.PSet(
        binOfMaximum = cms.int32(5),
        doPhotoStatistics = cms.bool(True),
        doSiPMSmearing = cms.bool(False),
        firstRing = cms.int32(1),
        readoutFrameSize = cms.int32(10),
        samplingFactors = cms.vdouble(
            231.0, 231.0, 231.0, 231.0, 360.0,
            360.0, 360.0, 360.0, 360.0, 360.0,
            360.0, 360.0, 360.0, 360.0, 360.0
        ),
        siPMCode = cms.int32(2),
        simHitToPhotoelectrons = cms.double(4000.0),
        sipmTau = cms.double(5.0),
        syncPhase = cms.bool(True),
        threshold_currentTDC = cms.double(-999.0),
        timePhase = cms.double(5.0),
        timeSmearing = cms.bool(False)
    ),
    ignoreGeantTime = cms.bool(False),
    injectTestHits = cms.bool(False),
    injectTestHitsCells = cms.vint32(),
    injectTestHitsEnergy = cms.vdouble(),
    injectTestHitsTime = cms.vdouble(),
    killHE = cms.bool(True),
    minFCToDelay = cms.double(5.0),
    zdc = cms.PSet(
        binOfMaximum = cms.int32(5),
        doPhotoStatistics = cms.bool(True),
        doSiPMSmearing = cms.bool(False),
        photoelectronsToAnalog = cms.double(1.843),
        readoutFrameSize = cms.int32(10),
        samplingFactor = cms.double(1.0),
        simHitToPhotoelectrons = cms.double(6.0),
        sipmTau = cms.double(0.0),
        syncPhase = cms.bool(True),
        threshold_currentTDC = cms.double(-999.0),
        timePhase = cms.double(-4.0)
    )
)