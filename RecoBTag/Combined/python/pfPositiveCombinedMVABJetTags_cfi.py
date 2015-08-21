import FWCore.ParameterSet.Config as cms

pfPositiveCombinedMVABJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidatePositiveCombinedMVAComputer'),
	tagInfos = cms.VInputTag(
		cms.InputTag("pfImpactParameterTagInfos"),
		cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfos"),
		cms.InputTag("softPFMuonsTagInfos"),
		cms.InputTag("softPFElectronsTagInfos")
	)
)

