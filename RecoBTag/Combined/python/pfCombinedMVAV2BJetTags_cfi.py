import FWCore.ParameterSet.Config as cms

pfCombinedMVAV2BJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidateCombinedMVAV2Computer'),
	tagInfos = cms.VInputTag(
		cms.InputTag("pfImpactParameterTagInfos"),
		cms.InputTag("pfSecondaryVertexTagInfos"),
		cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfos"),
		cms.InputTag("softPFMuonsTagInfos"),
		cms.InputTag("softPFElectronsTagInfos")
	)
)
