import FWCore.ParameterSet.Config as cms

negativeOnlyJetProbabilityJetTags = cms.EDProducer("JetTagProducer", 
     jetTagComputer = cms.string('negativeOnlyJetProbability'),
     tagInfos =  cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)
