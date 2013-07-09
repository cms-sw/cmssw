import FWCore.ParameterSet.Config as cms

negativeCombinedMVABJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('negativeCombinedMVA'),
	tagInfos = cms.VInputTag(
		cms.InputTag("impactParameterTagInfos"),
		cms.InputTag("secondaryVertexTagInfos"),
		cms.InputTag("softMuonTagInfos"),
		cms.InputTag("softElectronTagInfos")
	)
)
