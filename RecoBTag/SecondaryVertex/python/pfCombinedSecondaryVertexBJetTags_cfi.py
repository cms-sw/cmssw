import FWCore.ParameterSet.Config as cms

pfCombinedSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateCombinedSecondaryVertexComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), 
			     cms.InputTag("pfSecondaryVertexTagInfos"))
)
