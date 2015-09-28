import FWCore.ParameterSet.Config as cms

pfPositiveCombinedSecondaryVertexSoftLeptonCvsLJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidatePositiveCombinedSecondaryVertexSoftLeptonCvsLComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
	                         cms.InputTag("pfInclusiveSecondaryVertexFinderCvsLTagInfos"),
	                         cms.InputTag("softPFMuonsTagInfos"),
	                         cms.InputTag("softPFElectronsTagInfos"))
)
