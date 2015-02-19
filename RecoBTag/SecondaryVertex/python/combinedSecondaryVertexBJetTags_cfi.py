import FWCore.ParameterSet.Config as cms

combinedSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('combinedSecondaryVertexComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"),
	                         cms.InputTag("secondaryVertexTagInfos"))
)
