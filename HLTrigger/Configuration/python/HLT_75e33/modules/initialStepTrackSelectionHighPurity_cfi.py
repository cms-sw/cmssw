import FWCore.ParameterSet.Config as cms

initialStepTrackSelectionHighPurity = cms.EDProducer("TrackCollectionFilterCloner",
    copyExtras = cms.untracked.bool(True),
    copyTrajectories = cms.untracked.bool(False),
    minQuality = cms.string('highPurity'),
    originalMVAVals = cms.InputTag("initialStepTrackCutClassifier","MVAValues"),
    originalQualVals = cms.InputTag("initialStepTrackCutClassifier","QualityMasks"),
    originalSource = cms.InputTag("initialStepTracks")
)
