import FWCore.ParameterSet.Config as cms

pfCombinedSecondaryVertexSoftLeptonBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidateCombinedSecondaryVertexSoftLeptonComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
	                         cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfos"),
	                         cms.InputTag("softPFMuonsTagInfos"),
	                         cms.InputTag("softPFElectronsTagInfos"))
)
