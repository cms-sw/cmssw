import FWCore.ParameterSet.Config as cms

combinedSecondaryVertexNegativeBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('combinedSecondaryVertexNegative'),
	tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"),
	                         cms.InputTag("secondaryVertexNegativeTagInfos"))
)
