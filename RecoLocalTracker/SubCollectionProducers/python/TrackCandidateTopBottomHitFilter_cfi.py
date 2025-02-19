import FWCore.ParameterSet.Config as cms

trackCandidateTopBottomHitFilter = cms.EDProducer("TrackCandidateTopBottomHitFilter",
    Input = cms.InputTag("ckfTrackCandidatesP5Top"),
    TTRHBuilder = cms.string('WithoutRefit'),
    SeedY = cms.double(0)
)



