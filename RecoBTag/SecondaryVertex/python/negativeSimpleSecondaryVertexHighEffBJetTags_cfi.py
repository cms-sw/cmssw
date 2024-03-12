import FWCore.ParameterSet.Config as cms

negativeSimpleSecondaryVertexHighEffBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('simpleSecondaryVertex2TrkComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("secondaryVertexNegativeTagInfos"))
)
# foo bar baz
# Q0VUHUxu1uBlG
# W8rAiOK05bJbU
