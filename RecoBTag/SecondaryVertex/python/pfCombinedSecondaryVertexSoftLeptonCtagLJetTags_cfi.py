import FWCore.ParameterSet.Config as cms

pfCombinedSecondaryVertexSoftLeptonCtagLJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidateCombinedSecondaryVertexSoftLeptonCtagLComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
	                         cms.InputTag("pfInclusiveSecondaryVertexFinderCtagLTagInfos"),
	                         cms.InputTag("softPFMuonsTagInfos"),
	                         cms.InputTag("softPFElectronsTagInfos"))
)
