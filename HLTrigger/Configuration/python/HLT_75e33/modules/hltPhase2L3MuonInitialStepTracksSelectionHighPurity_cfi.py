import FWCore.ParameterSet.Config as cms

hltPhase2L3MuonInitialStepTracksSelectionHighPurity = cms.EDProducer("TrackCollectionFilterCloner",
    copyExtras = cms.untracked.bool(True),
    copyTrajectories = cms.untracked.bool(False),
    minQuality = cms.string('highPurity'),
    originalMVAVals = cms.InputTag("hltPhase2L3MuonInitialStepTrackCutClassifier","MVAValues"),
    originalQualVals = cms.InputTag("hltPhase2L3MuonInitialStepTrackCutClassifier","QualityMasks"),
    originalSource = cms.InputTag("hltPhase2L3MuonInitialStepTracks")
)
