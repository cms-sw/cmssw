import FWCore.ParameterSet.Config as cms

hltPhase2MuonIOTrackMkFitSeeds = cms.EDProducer("MkFitSeedConverter",
        maxNSeeds = cms.uint32(500000),
        seeds = cms.InputTag("hltPhase2MuonIOTrackSeeds"),
        ttrhBuilder = cms.ESInputTag("","hltESPTTRHBuilderWithTrackAngle")
)
