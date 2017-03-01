import FWCore.ParameterSet.Config as cms

pfGhostTrackBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidateGhostTrackComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
	                         cms.InputTag("pfGhostTrackVertexTagInfos"))
)
