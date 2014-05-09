import FWCore.ParameterSet.Config as cms

combinedSecondaryVertexSoftLeptonBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('combinedSecondaryVertexSoftLepton'),
	tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"),
	                         cms.InputTag("secondaryVertexTagInfosV2"),
	                         cms.InputTag("softPFMuonsTagInfos"),
	                         cms.InputTag("softPFElectronsTagInfos")
													 )
)
