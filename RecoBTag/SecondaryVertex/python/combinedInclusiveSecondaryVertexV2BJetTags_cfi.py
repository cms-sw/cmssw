import FWCore.ParameterSet.Config as cms

combinedInclusiveSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('combinedSecondaryVertexV2Computer'),
	tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"),
	                         cms.InputTag("inclusiveSecondaryVertexFinderTagInfos"))
)
# foo bar baz
# CPU6XG3xXN4Bo
# OQ04asB6EUT2c
