import FWCore.ParameterSet.Config as cms

hltInitialStepTrackFeatureExtractor = cms.EDProducer("alpaka_serial_sync::TrackFeatureExtractor",
    src = cms.InputTag("hltInitialStepTracks"),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    )
)
