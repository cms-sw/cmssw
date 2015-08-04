import FWCore.ParameterSet.Config as cms

pfCombinedMVABJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidateCombinedMVAComputer'),
	tagInfos = cms.VInputTag(
		cms.InputTag("pfImpactParameterTagInfos"),
		cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfos"),
		cms.InputTag("softPFMuonsTagInfos"),
		cms.InputTag("softPFElectronsTagInfos")
	)
)
