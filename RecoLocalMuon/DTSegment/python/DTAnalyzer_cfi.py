import FWCore.ParameterSet.Config as cms

DTAnalyzer = cms.EDAnalyzer("DTAnalyzer",
    LCT_DT = cms.bool(True),
    DTLocalTriggerLabel = cms.string('dtunpacker'),
    recHits2DLabel = cms.string('dt2DSegments'),
    LCT_CSC = cms.bool(False),
    LCT_RPC = cms.bool(False),
    recHits4DLabel = cms.string('dt4DSegments'),
    debug = cms.untracked.bool(False),
    rootFileName = cms.untracked.string('DTAnalyzer.root'),
    SALabel = cms.string('CosmicMuon'),
    isMC = cms.bool(False),
    tTrigModeConfig = cms.untracked.PSet(
        vPropWire = cms.double(24.4),
        doTOFCorrection = cms.bool(False),
        kFactor = cms.double(-1.3),
        doWirePropCorrection = cms.bool(False),
        doT0Correction = cms.bool(True),
        debug = cms.untracked.bool(False)
    ),
    tTrigMode = cms.untracked.string('DTTTrigSyncFromDB'),
    recHits1DLabel = cms.string('dt1DRecHits')
)



