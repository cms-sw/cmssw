import FWCore.ParameterSet.Config as cms

combinedSecondaryVertexNegativeBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('combinedSecondaryVertexNegative'),
	tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"),
	                         cms.InputTag("secondaryVertexNegativeTagInfos"))
)

combinedSecondaryVertexV1NegativeBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('combinedSecondaryVertexV1Negative'),
	tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"),
	                         cms.InputTag("secondaryVertexNegativeTagInfos"))
)
