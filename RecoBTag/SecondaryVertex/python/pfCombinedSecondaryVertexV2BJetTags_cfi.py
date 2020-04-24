import FWCore.ParameterSet.Config as cms

pfCombinedSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateCombinedSecondaryVertexV2Computer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), 
			     cms.InputTag("pfSecondaryVertexTagInfos"))
)
