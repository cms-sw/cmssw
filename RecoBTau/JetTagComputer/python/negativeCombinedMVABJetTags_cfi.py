import FWCore.ParameterSet.Config as cms

negativeCombinedMVABJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('negativeCombinedMVA'),
	tagInfos = cms.VInputTag(
		cms.InputTag("impactParameterTagInfos"),
		cms.InputTag("secondaryVertexNegativeTagInfos"),
		cms.InputTag("softPFMuonsTagInfos"),
		cms.InputTag("softPFElectronsTagInfos")
	)
)

