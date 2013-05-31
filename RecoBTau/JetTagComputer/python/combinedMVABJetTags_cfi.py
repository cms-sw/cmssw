import FWCore.ParameterSet.Config as cms

combinedMVABJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('combinedMVA'),
	tagInfos = cms.VInputTag(
		cms.InputTag("impactParameterTagInfos"),
		cms.InputTag("secondaryVertexTagInfos"),
		cms.InputTag("softMuonTagInfos"),
		cms.InputTag("softElectronTagInfos")
	)
)

combinedSecondaryVertexSoftPFLeptonV1BJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('combinedSecondaryVertexSoftPFLeptonV1'),
	tagInfos = cms.VInputTag(
		cms.InputTag("impactParameterTagInfos"),
		cms.InputTag("secondaryVertexTagInfos"),
		cms.InputTag("softPFMuonsTagInfos"),
		cms.InputTag("softPFElectronsTagInfos")
	)
)
