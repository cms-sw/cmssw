import FWCore.ParameterSet.Config as cms

trackCandidateTopBottomHitFilter = cms.EDFilter("TrackCandidateTopBottomHitFilter",
    Input = cms.InputTag("ckfTrackCandidatesP5Top"),
    TTRHBuilder = cms.string('WithoutRefit'),
    SeedY = cms.double(0)
)



