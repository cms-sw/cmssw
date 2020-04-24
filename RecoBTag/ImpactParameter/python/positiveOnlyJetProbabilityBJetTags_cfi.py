import FWCore.ParameterSet.Config as cms

positiveOnlyJetProbabilityBJetTags = cms.EDProducer("JetTagProducer",  
     jetTagComputer = cms.string('positiveOnlyJetProbabilityComputer'),
     tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)
