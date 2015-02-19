import FWCore.ParameterSet.Config as cms

negativeCombinedSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('negativeCombinedSecondaryVertexComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"),
	                         cms.InputTag("secondaryVertexNegativeTagInfos"))
)

