import FWCore.ParameterSet.Config as cms

initialStepTrackSelectionHighPurity = cms.EDProducer("TrackCollectionFilterCloner",
    copyExtras = cms.untracked.bool(True),
    copyTrajectories = cms.untracked.bool(False),
    minQuality = cms.string('highPurity'),
    originalMVAVals = cms.InputTag("initialStepTrackCutClassifier","MVAValues"),
    originalQualVals = cms.InputTag("initialStepTrackCutClassifier","QualityMasks"),
    originalSource = cms.InputTag("initialStepTracks")
)
# foo bar baz
# 2AdekGk9djrMA
# wkeqZ4Yp1ItZa
