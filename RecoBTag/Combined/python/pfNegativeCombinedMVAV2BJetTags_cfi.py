import FWCore.ParameterSet.Config as cms

pfNegativeCombinedMVAV2BJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidateNegativeCombinedMVAV2Computer'),
	tagInfos = cms.VInputTag(
		cms.InputTag("pfImpactParameterTagInfos"),
		cms.InputTag("pfSecondaryVertexNegativeTagInfos"),
		cms.InputTag("pfInclusiveSecondaryVertexFinderNegativeTagInfos"),
		cms.InputTag("softPFMuonsTagInfos"),
		cms.InputTag("softPFElectronsTagInfos")
	)
)
