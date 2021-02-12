import FWCore.ParameterSet.Config as cms

hltPfJetProbabilityBJetTagsPuppi = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('hltCandidateJetProbabilityComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("hltDeepBLifetimeTagInfosPFPuppi"))
)
