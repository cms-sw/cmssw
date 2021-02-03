import FWCore.ParameterSet.Config as cms

dt4DCosmicSegments = cms.EDProducer("DTRecSegment4DProducer",
    Reco4DAlgoConfig = cms.PSet(
        AllDTRecHits = cms.bool(True),
        Reco2DAlgoConfig = cms.PSet(
            AlphaMaxPhi = cms.double(100.0),
            AlphaMaxTheta = cms.double(100.0),
            MaxAllowedHits = cms.uint32(50),
            MaxChi2 = cms.double(4.0),
            debug = cms.untracked.bool(False),
            hit_afterT0_resolution = cms.double(0.03),
            intime_cut = cms.double(-1.0),
            nSharedHitsMax = cms.int32(2),
            nUnSharedHitsMin = cms.int32(2),
            performT0SegCorrection = cms.bool(False),
            performT0_vdriftSegCorrection = cms.bool(False),
            perform_delta_rejecting = cms.bool(False),
            recAlgo = cms.string('DTLinearDriftFromDBAlgo'),
            recAlgoConfig = cms.PSet(
                debug = cms.untracked.bool(False),
                doVdriftCorr = cms.bool(False),
                maxTime = cms.double(420.0),
                minTime = cms.double(-3.0),
                stepTwoFromDigi = cms.bool(False),
                tTrigMode = cms.string('DTTTrigSyncFromDB'),
                tTrigModeConfig = cms.PSet(
                    debug = cms.untracked.bool(False),
                    doT0Correction = cms.bool(True),
                    doTOFCorrection = cms.bool(False),
                    doWirePropCorrection = cms.bool(False),
                    tTrigLabel = cms.string('cosmics'),
                    tofCorrType = cms.int32(0),
                    vPropWire = cms.double(24.4),
                    wirePropCorrType = cms.int32(0)
                ),
                useUncertDB = cms.bool(False)
            ),
            segmCleanerMode = cms.int32(2)
        ),
        Reco2DAlgoName = cms.string('DTMeantimerPatternReco'),
        debug = cms.untracked.bool(False),
        hit_afterT0_resolution = cms.double(0.03),
        intime_cut = cms.double(-1.0),
        nUnSharedHitsMin = cms.int32(2),
        performT0SegCorrection = cms.bool(False),
        performT0_vdriftSegCorrection = cms.bool(False),
        perform_delta_rejecting = cms.bool(False),
        recAlgo = cms.string('DTLinearDriftFromDBAlgo'),
        recAlgoConfig = cms.PSet(
            debug = cms.untracked.bool(False),
            doVdriftCorr = cms.bool(False),
            maxTime = cms.double(420.0),
            minTime = cms.double(-3.0),
            stepTwoFromDigi = cms.bool(False),
            tTrigMode = cms.string('DTTTrigSyncFromDB'),
            tTrigModeConfig = cms.PSet(
                debug = cms.untracked.bool(False),
                doT0Correction = cms.bool(True),
                doTOFCorrection = cms.bool(False),
                doWirePropCorrection = cms.bool(False),
                tTrigLabel = cms.string('cosmics'),
                tofCorrType = cms.int32(0),
                vPropWire = cms.double(24.4),
                wirePropCorrType = cms.int32(0)
            ),
            useUncertDB = cms.bool(False)
        )
    ),
    Reco4DAlgoName = cms.string('DTMeantimerPatternReco4D'),
    debug = cms.untracked.bool(False),
    recHits1DLabel = cms.InputTag("dt1DCosmicRecHits"),
    recHits2DLabel = cms.InputTag("dt2DCosmicSegments")
)
