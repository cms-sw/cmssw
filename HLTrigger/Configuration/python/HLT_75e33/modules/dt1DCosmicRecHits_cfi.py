import FWCore.ParameterSet.Config as cms

dt1DCosmicRecHits = cms.EDProducer("DTRecHitProducer",
    debug = cms.untracked.bool(False),
    dtDigiLabel = cms.InputTag("muonDTDigis"),
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
)
