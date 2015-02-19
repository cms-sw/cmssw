import FWCore.ParameterSet.Config as cms

positiveCombinedSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('positiveCombinedSecondaryVertexComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"),
	                         cms.InputTag("secondaryVertexTagInfos"))
)
