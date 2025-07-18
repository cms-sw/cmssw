import FWCore.ParameterSet.Config as cms

hltHighPtTripletStepTrackSelectionHighPurity = cms.EDProducer("TrackCollectionFilterCloner",
    copyExtras = cms.untracked.bool(True),
    copyTrajectories = cms.untracked.bool(False),
    minQuality = cms.string('highPurity'),
    originalMVAVals = cms.InputTag("hltHighPtTripletStepTrackCutClassifier","MVAValues"),
    originalQualVals = cms.InputTag("hltHighPtTripletStepTrackCutClassifier","QualityMasks"),
    originalSource = cms.InputTag("hltHighPtTripletStepTracks")
)
