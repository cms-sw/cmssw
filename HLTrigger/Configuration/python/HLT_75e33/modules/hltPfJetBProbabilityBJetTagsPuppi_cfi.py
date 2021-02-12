import FWCore.ParameterSet.Config as cms

hltPfJetBProbabilityBJetTagsPuppi = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('hltCandidateJetBProbabilityComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("hltDeepBLifetimeTagInfosPFPuppi"))
)
