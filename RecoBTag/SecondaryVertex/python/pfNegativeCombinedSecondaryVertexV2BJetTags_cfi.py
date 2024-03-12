import FWCore.ParameterSet.Config as cms

pfNegativeCombinedSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidateNegativeCombinedSecondaryVertexV2Computer'),
	tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
	                         cms.InputTag("pfSecondaryVertexNegativeTagInfos"))
)
# foo bar baz
# lFADLfBl6Uwco
# LOmDTQS5jg9gR
