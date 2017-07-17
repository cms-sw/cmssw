import FWCore.ParameterSet.Config as cms

ghostTrackBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('ghostTrackComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"),
	                         cms.InputTag("ghostTrackVertexTagInfos"))
)
