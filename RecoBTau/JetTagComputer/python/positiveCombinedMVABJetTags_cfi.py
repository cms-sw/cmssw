import FWCore.ParameterSet.Config as cms

positiveCombinedMVABJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('positiveCombinedMVA'),
	tagInfos = cms.VInputTag(
		cms.InputTag("impactParameterTagInfos"),
		cms.InputTag("secondaryVertexTagInfos"),
		cms.InputTag("softPFMuonsTagInfos"),
		cms.InputTag("softPFElectronsTagInfos")
	)
)

positiveCombinedSecondaryVertexSoftPFLeptonV1BJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('positiveCombinedSecondaryVertexSoftPFLeptonV1'),
	tagInfos = cms.VInputTag(
		cms.InputTag("impactParameterTagInfos"),
		cms.InputTag("secondaryVertexTagInfos"),
		cms.InputTag("softPFMuonsTagInfos"),
		cms.InputTag("softPFElectronsTagInfos")
	)
)
