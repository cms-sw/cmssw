import FWCore.ParameterSet.Config as cms

pfNegativeCombinedSecondaryVertexSoftLeptonCtagLJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidateNegativeCombinedSecondaryVertexSoftLeptonCtagLComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
	                         cms.InputTag("pfInclusiveSecondaryVertexFinderCtagLNegativeTagInfos"),
	                         cms.InputTag("softPFMuonsTagInfos"),
	                         cms.InputTag("softPFElectronsTagInfos"))
)
