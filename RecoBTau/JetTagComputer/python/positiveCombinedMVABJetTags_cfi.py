import FWCore.ParameterSet.Config as cms

positiveCombinedMVABJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('positiveCombinedMVA'),
	tagInfos = cms.VInputTag(
		cms.InputTag("impactParameterTagInfos"),
		cms.InputTag("secondaryVertexTagInfos"),
		cms.InputTag("softMuonTagInfos"),
		cms.InputTag("softElectronTagInfos")
	)
)
