import FWCore.ParameterSet.Config as cms

combinedMVAV2BJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('combinedMVAV2Computer'),
	tagInfos = cms.VInputTag(
		cms.InputTag("impactParameterTagInfos"),
		cms.InputTag("secondaryVertexTagInfos"),
		cms.InputTag("inclusiveSecondaryVertexFinderTagInfos"),
		cms.InputTag("softPFMuonsTagInfos"),
		cms.InputTag("softPFElectronsTagInfos")
	)
)
