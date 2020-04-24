import FWCore.ParameterSet.Config as cms

negativeOnlyJetProbabilityBJetTags = cms.EDProducer("JetTagProducer", 
    jetTagComputer = cms.string('negativeOnlyJetProbabilityComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)
