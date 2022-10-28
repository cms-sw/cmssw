import FWCore.ParameterSet.Config as cms

highPtTripletStepTrackSelectionHighPurity = cms.EDProducer("TrackCollectionFilterCloner",
    copyExtras = cms.untracked.bool(True),
    copyTrajectories = cms.untracked.bool(False),
    minQuality = cms.string('highPurity'),
    originalMVAVals = cms.InputTag("highPtTripletStepTrackCutClassifier","MVAValues"),
    originalQualVals = cms.InputTag("highPtTripletStepTrackCutClassifier","QualityMasks"),
    originalSource = cms.InputTag("highPtTripletStepTracks")
)
