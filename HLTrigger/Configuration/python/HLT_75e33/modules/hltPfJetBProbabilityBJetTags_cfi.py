import FWCore.ParameterSet.Config as cms

hltPfJetBProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('hltCandidateJetBProbabilityComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("hltDeepBLifetimeTagInfosPF"))
)
