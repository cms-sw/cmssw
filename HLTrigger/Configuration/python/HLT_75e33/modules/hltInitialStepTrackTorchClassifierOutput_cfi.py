import FWCore.ParameterSet.Config as cms

hltInitialStepTrackTorchClassifierOutput = cms.EDProducer("TrackTorchClassifierFromSoA",
    src = cms.InputTag("hltInitialStepTracks"),
    scores = cms.InputTag("hltInitialStepTrackTorchClassifier"),
    features = cms.InputTag("hltInitialStepTrackFeatureExtractor"),
    minScore = cms.double(0.377),
    dxyThreshold = cms.double(0.5),
    highDxyMinScore = cms.double(0.267)
)
