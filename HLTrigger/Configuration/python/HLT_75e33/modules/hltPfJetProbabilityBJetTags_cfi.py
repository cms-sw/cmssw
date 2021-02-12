import FWCore.ParameterSet.Config as cms

hltPfJetProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('hltCandidateJetProbabilityComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("hltDeepBLifetimeTagInfosPF"))
)
