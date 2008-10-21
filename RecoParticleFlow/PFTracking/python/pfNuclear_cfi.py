import FWCore.ParameterSet.Config as cms

pfNuclear = cms.EDProducer("PFNuclearProducer",
    # cut on the likelihood of the nuclear interaction
    likelihoodCut = cms.double(0.1),
    nuclearColList = cms.VInputTag(cms.InputTag("firstnuclearInteractionMaker"), cms.InputTag("secondnuclearInteractionMaker"), cms.InputTag("thirdnuclearInteractionMaker"), cms.InputTag("fourthnuclearInteractionMaker"))
)


