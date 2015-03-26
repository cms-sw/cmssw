import FWCore.ParameterSet.Config as cms

pfPositiveCombinedSecondaryVertexSoftLeptonBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidatePositiveCombinedSecondaryVertexSoftLeptonComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
	                         cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfos"),
	                         cms.InputTag("softPFMuonsTagInfos"),
	                         cms.InputTag("softPFElectronsTagInfos"))
)
