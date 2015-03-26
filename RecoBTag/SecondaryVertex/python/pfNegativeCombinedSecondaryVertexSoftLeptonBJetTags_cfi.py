import FWCore.ParameterSet.Config as cms

pfNegativeCombinedSecondaryVertexSoftLeptonBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidateNegativeCombinedSecondaryVertexSoftLeptonComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
	                         cms.InputTag("pfInclusiveSecondaryVertexFinderNegativeTagInfos"),
	                         cms.InputTag("softPFMuonsTagInfos"),
	                         cms.InputTag("softPFElectronsTagInfos"))
)
