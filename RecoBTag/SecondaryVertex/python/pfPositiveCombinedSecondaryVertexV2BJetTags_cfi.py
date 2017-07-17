import FWCore.ParameterSet.Config as cms

pfPositiveCombinedSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidatePositiveCombinedSecondaryVertexV2Computer'),
	tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
	                         cms.InputTag("pfSecondaryVertexTagInfos"))
)
