import FWCore.ParameterSet.Config as cms

combinedSecondaryVertexSoftLeptonBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('combinedSecondaryVertexSoftLeptonComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"),
	                         cms.InputTag("secondaryVertexTagInfos"),
	                         cms.InputTag("softPFMuonsTagInfos"),
	                         cms.InputTag("softPFElectronsTagInfos"))
)
