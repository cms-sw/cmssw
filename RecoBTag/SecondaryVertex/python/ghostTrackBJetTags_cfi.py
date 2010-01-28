import FWCore.ParameterSet.Config as cms

ghostTrackBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('ghostTrack'),
	tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"),
	                         cms.InputTag("ghostTrackVertexTagInfos"))
)
