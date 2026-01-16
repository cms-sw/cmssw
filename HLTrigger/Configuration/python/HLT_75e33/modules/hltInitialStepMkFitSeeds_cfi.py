import FWCore.ParameterSet.Config as cms

# MkFitSeedConverter options
hltInitialStepMkFitSeeds = cms.EDProducer("MkFitSeedConverter",
        maxNSeeds = cms.uint32(500000),
        mightGet = cms.optional.untracked.vstring,
        seeds = cms.InputTag("hltInitialStepSeeds"),
        ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
)

from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
from Configuration.ProcessModifiers.seedingLST_cff import seedingLST
from Configuration.ProcessModifiers.hltTrackingMkFitInitialStep_cff import hltTrackingMkFitInitialStep
(trackingLST & seedingLST & hltTrackingMkFitInitialStep).toModify(hltInitialStepMkFitSeeds, seeds = "hltInitialStepTrajectorySeedsLST")
