import FWCore.ParameterSet.Config as cms

simpleInclusiveSecondaryVertexHighPurBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('simpleSecondaryVertex3TrkComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("inclusiveSecondaryVertexFinderFilteredTagInfos"))
)
# foo bar baz
# a4p0qPxqtStmb
# XZwyMqa8W8z3Y
