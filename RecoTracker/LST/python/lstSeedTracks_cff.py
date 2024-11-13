import FWCore.ParameterSet.Config as cms

lstInitialStepSeedTracks = cms.EDProducer(
    "TrackFromSeedProducer",
    src = cms.InputTag("initialStepSeeds"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    TTRHBuilder = cms.string("WithoutRefit")
)

lstHighPtTripletStepSeedTracks = cms.EDProducer(
    "TrackFromSeedProducer",
    src = cms.InputTag("highPtTripletStepSeeds"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    TTRHBuilder = cms.string("WithoutRefit")
)
