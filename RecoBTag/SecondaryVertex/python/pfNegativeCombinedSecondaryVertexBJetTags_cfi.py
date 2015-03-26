import FWCore.ParameterSet.Config as cms

pfNegativeCombinedSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidateNegativeCombinedSecondaryVertexComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
	                         cms.InputTag("pfSecondaryVertexNegativeTagInfos"))
)
