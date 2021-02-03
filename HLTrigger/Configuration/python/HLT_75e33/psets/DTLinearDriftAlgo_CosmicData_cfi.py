import FWCore.ParameterSet.Config as cms

DTLinearDriftAlgo_CosmicData = cms.PSet(
    recAlgo = cms.string('DTLinearDriftAlgo'),
    recAlgoConfig = cms.PSet(
        debug = cms.untracked.bool(False),
        driftVelocity = cms.double(0.00543),
        hitResolution = cms.double(0.02),
        maxTime = cms.double(420.0),
        minTime = cms.double(-3.0),
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
        )
    )
)