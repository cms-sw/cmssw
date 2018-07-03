import FWCore.ParameterSet.Config as cms

negativeCombinedMVAV2BJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('negativeCombinedMVAV2Computer'),
	tagInfos = cms.VInputTag(
		cms.InputTag("impactParameterTagInfos"),
		cms.InputTag("secondaryVertexNegativeTagInfos"),
		cms.InputTag("inclusiveSecondaryVertexFinderNegativeTagInfos"),
		cms.InputTag("softPFMuonsTagInfos"),
		cms.InputTag("softPFElectronsTagInfos")
	)
)
