import FWCore.ParameterSet.Config as cms

DTSegAnalyzer = cms.EDAnalyzer("DTSegAnalyzer",
    doHits = cms.bool(True),
    tTrigMode = cms.untracked.string('DTTTrigSyncFromDB'),
    recHits2DLabel = cms.string('dt2DSegments'),
    doSegs = cms.bool(True),
    doSA = cms.bool(True),
    recHits4DLabel = cms.string('dt4DSegments'),
    rootFileName = cms.untracked.string('DTSegAnalyzer.root'),
    debug = cms.untracked.bool(False),
    tTrigModeConfig = cms.untracked.PSet(
        vPropWire = cms.double(24.4),
        doTOFCorrection = cms.bool(False),
        tofCorrType = cms.int32(2),
        kFactor = cms.double(-1.3),
        wirePropCorrType = cms.int32(0),
        doWirePropCorrection = cms.bool(False),
        doT0Correction = cms.bool(True),
        debug = cms.untracked.bool(False)
    ),
    recHits1DLabel = cms.string('dt1DRecHits')
)



