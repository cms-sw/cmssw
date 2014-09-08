import FWCore.ParameterSet.Config as cms

combinedMVABJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('combinedMVA'),
	tagInfos = cms.VInputTag(
		cms.InputTag("impactParameterTagInfos"),
		cms.InputTag("secondaryVertexTagInfos"),
		cms.InputTag("softPFMuonsTagInfos"),
		cms.InputTag("softPFElectronsTagInfos")
	)
)
