import FWCore.ParameterSet.Config as cms

positiveOnlyJetProbabilityJetTags = cms.EDProducer("JetTagProducer",  
     jetTagComputer = cms.string('positiveOnlyJetProbability'),
     tagInfos =  cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)


negativeOnlyJetProbabilityJetTags = cms.EDProducer("JetTagProducer", 
     jetTagComputer = cms.string('negativeOnlyJetProbability'),
     tagInfos =  cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)

negativeTrackCountingHigEffJetTags = cms.EDProducer("JetTagProducer", 
     jetTagComputer = cms.string('negativeTrackCounting3D2nd'),
     tagInfos =  cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)


negativeTrackCountingHigPur = cms.EDProducer("JetTagProducer",
     jetTagComputer = cms.string('negativeTrackCounting3D3rd'),
     tagInfos =  cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)

