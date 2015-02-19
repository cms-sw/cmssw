import FWCore.ParameterSet.Config as cms

pfPositiveCombinedSecondaryVertexSoftLeptonCtagLJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidatePositiveCombinedSecondaryVertexSoftLeptonCtagLComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
	                         cms.InputTag("pfInclusiveSecondaryVertexFinderCtagLTagInfos"),
	                         cms.InputTag("softPFMuonsTagInfos"),
	                         cms.InputTag("softPFElectronsTagInfos"))
)
