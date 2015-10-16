import FWCore.ParameterSet.Config as cms

negativeCombinedMVABJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('negativeCombinedMVAComputer'),
	tagInfos = cms.VInputTag(
		cms.InputTag("impactParameterTagInfos"),
		cms.InputTag("inclusiveSecondaryVertexFinderNegativeTagInfos"),
		cms.InputTag("softPFMuonsTagInfos"),
		cms.InputTag("softPFElectronsTagInfos")
	)
)

