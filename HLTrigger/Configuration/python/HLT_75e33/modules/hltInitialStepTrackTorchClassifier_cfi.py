import FWCore.ParameterSet.Config as cms

hltInitialStepTrackTorchClassifier = cms.EDProducer("alpaka_serial_sync::TrackTorchClassifierAlpaka",
    modelPath = cms.FileInPath('RecoTracker/FinalTrackSelectors/data/TrackTorchClassifier/model.pt'),
    features = cms.InputTag("hltInitialStepTrackFeatureExtractor"),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    )
)
