import FWCore.ParameterSet.Config as cms

hltPhase2L3MuonHighPtTripletStepTracksSelectionHighPurity = cms.EDProducer("TrackCollectionFilterCloner",
    copyExtras = cms.untracked.bool(True),
    copyTrajectories = cms.untracked.bool(False),
    minQuality = cms.string('highPurity'),
    originalMVAVals = cms.InputTag("hltPhase2L3MuonHighPtTripletStepTrackCutClassifier","MVAValues"),
    originalQualVals = cms.InputTag("hltPhase2L3MuonHighPtTripletStepTrackCutClassifier","QualityMasks"),
    originalSource = cms.InputTag("hltPhase2L3MuonHighPtTripletStepTracks")
)
