import FWCore.ParameterSet.Config as cms

positiveCombinedMVABJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('positiveCombinedMVAComputer'),
	tagInfos = cms.VInputTag(
		cms.InputTag("impactParameterTagInfos"),
		cms.InputTag("inclusiveSecondaryVertexFinderTagInfos"),
		cms.InputTag("softPFMuonsTagInfos"),
		cms.InputTag("softPFElectronsTagInfos")
	)
)

