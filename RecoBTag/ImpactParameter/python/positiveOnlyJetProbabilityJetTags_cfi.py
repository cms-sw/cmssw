import FWCore.ParameterSet.Config as cms

positiveOnlyJetProbabilityJetTags = cms.EDProducer("JetTagProducer",  
     jetTagComputer = cms.string('positiveOnlyJetProbability'),
     tagInfos =  cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)
