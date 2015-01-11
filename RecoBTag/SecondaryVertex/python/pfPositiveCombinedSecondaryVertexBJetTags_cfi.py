import FWCore.ParameterSet.Config as cms

pfPositiveCombinedSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidatePositiveCombinedSecondaryVertexComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
	                         cms.InputTag("pfSecondaryVertexTagInfos"))
)
