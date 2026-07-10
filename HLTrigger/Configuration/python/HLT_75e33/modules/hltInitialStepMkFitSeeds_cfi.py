import FWCore.ParameterSet.Config as cms

# MkFitSeedConverter options
hltInitialStepMkFitSeeds = cms.EDProducer("MkFitSeedConverter",
        maxNSeeds = cms.uint32(500000),
        mightGet = cms.optional.untracked.vstring,
        seeds = cms.InputTag("hltInitialStepTrajectorySeedsLST"),
        ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
)
