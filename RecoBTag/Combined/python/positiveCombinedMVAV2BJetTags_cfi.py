import FWCore.ParameterSet.Config as cms

positiveCombinedMVAV2BJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('positiveCombinedMVAV2Computer'),
	tagInfos = cms.VInputTag(
		cms.InputTag("impactParameterTagInfos"),
		cms.InputTag("secondaryVertexPositiveTagInfos"),
		cms.InputTag("inclusiveSecondaryVertexFinderPositiveTagInfos"),
		cms.InputTag("softPFMuonsTagInfos"),
		cms.InputTag("softPFElectronsTagInfos")
	)
)
