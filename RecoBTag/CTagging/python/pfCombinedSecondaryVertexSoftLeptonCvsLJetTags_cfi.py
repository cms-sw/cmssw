import FWCore.ParameterSet.Config as cms

pfCombinedSecondaryVertexSoftLeptonCvsLJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidateCombinedSecondaryVertexSoftLeptonCvsLComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
	                         cms.InputTag("pfInclusiveSecondaryVertexFinderCvsLTagInfos"),
	                         cms.InputTag("softPFMuonsTagInfos"),
	                         cms.InputTag("softPFElectronsTagInfos"))
)
