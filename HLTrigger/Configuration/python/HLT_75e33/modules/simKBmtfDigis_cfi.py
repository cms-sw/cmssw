import FWCore.ParameterSet.Config as cms

simKBmtfDigis = cms.EDProducer("L1TMuonBarrelKalmanTrackProducer",
    algoSettings = cms.PSet(
        aPhi = cms.vdouble(1.942, 0.03125, 0.0273438, 0.015625),
        aPhiB = cms.vdouble(-1.508, -0.123047, -0.174805, -0.144531),
        aPhiBNLO = cms.vdouble(0.000331, 0, 0, 0),
        bPhi = cms.vdouble(-1, 0.154297, 0.173828, 0.147461),
        bPhiB = cms.vdouble(-1, 1.15332, 1.17285, 1.14648),
        chiSquare = cms.vdouble(0.0, 0.109375, 0.234375, 0.359375),
        chiSquareCut = cms.vint32(126, 126, 126, 126, 126),
        chiSquareCutCurvMax = cms.vint32(2500, 2500, 2500, 2500, 2500),
        chiSquareCutPattern = cms.vint32(7, 11, 13, 14, 15),
        chiSquareCutTight = cms.vint32(
            40, 126, 60, 126, 126,
            126
        ),
        combos1 = cms.vint32(),
        combos2 = cms.vint32(3),
        combos3 = cms.vint32(5, 6, 7),
        combos4 = cms.vint32(
            9, 10, 11, 12, 13,
            14, 15
        ),
        eLoss = cms.vdouble(0.000765, 0, 0, 0),
        etaLUT0 = cms.vdouble(8.946, 7.508, 6.279, 6.399),
        etaLUT1 = cms.vdouble(0.159, 0.116, 0.088, 0.128),
        initialK = cms.vdouble(-1.196, -1.581, -2.133, -2.263),
        initialK2 = cms.vdouble(-0.000326, -0.0007165, 0.002305, -0.00563),
        lutFile = cms.string('L1Trigger/L1TMuon/data/bmtf_luts/kalmanLUTs.root'),
        mScatteringPhi = cms.vdouble(0.00249, 5.47e-05, 3.49e-05, 1.37e-05),
        mScatteringPhiB = cms.vdouble(0.00722, 0.003461, 0.004447, 0.00412),
        phiAt2 = cms.double(0.15918),
        pointResolutionPhi = cms.double(1.0),
        pointResolutionPhiB = cms.double(500.0),
        pointResolutionVertex = cms.double(1.0),
        trackComp = cms.vdouble(1.75, 1.25, 0.625, 0.25),
        trackCompCut = cms.vint32(
            15, 15, 15, 15, 15,
            15
        ),
        trackCompCutCurvMax = cms.vint32(
            34, 34, 34, 34, 34,
            34
        ),
        trackCompCutPattern = cms.vint32(
            3, 5, 6, 9, 10,
            12
        ),
        trackCompErr1 = cms.vdouble(2.0, 2.0, 2.0, 2.0),
        trackCompErr2 = cms.vdouble(0.21875, 0.21875, 0.21875, 0.3125),
        useOfflineAlgo = cms.bool(False),
        verbose = cms.bool(False)
    ),
    bx = cms.vint32(-2, -1, 0, 1, 2),
    src = cms.InputTag("simKBmtfStubs"),
    trackFinderSettings = cms.PSet(
        sectorSettings = cms.PSet(
            regionSettings = cms.PSet(
                verbose = cms.int32(0)
            ),
            verbose = cms.int32(0),
            wheelsToProcess = cms.vint32(-2, -1, 0, 1, 2)
        ),
        sectorsToProcess = cms.vint32(
            0, 1, 2, 3, 4,
            5, 6, 7, 8, 9,
            10, 11
        ),
        verbose = cms.int32(0)
    )
)
