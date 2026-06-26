import FWCore.ParameterSet.Config as cms

hltInitialStepTrackFeatureExtractor = cms.EDProducer("TrackFeatureExtractor@alpaka",
    src = cms.InputTag("hltInitialStepTracks"),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    )
)
