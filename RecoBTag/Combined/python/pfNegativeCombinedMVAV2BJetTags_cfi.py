import FWCore.ParameterSet.Config as cms

pfNegativeCombinedMVAV2BJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidateNegativeCombinedMVAV2Computer'),
	tagInfos = cms.VInputTag(
		cms.InputTag("pfImpactParameterTagInfos"),
		cms.InputTag("pfSecondaryVertexTagInfos"),
		cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfos"),
		cms.InputTag("softPFMuonsTagInfos"),
		cms.InputTag("softPFElectronsTagInfos")
	)
)
