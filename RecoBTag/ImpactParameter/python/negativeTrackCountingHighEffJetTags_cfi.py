import FWCore.ParameterSet.Config as cms

negativeTrackCountingHighEffJetTags = cms.EDProducer("JetTagProducer", 
     jetTagComputer = cms.string('negativeTrackCounting3D2nd'),
     tagInfos =  cms.VInputTag(cms.InputTag("impactParameterTagInfos")),
     trackQualityClass = cms.string ( "any" )
)
