import FWCore.ParameterSet.Config as cms

negativeSimpleSecondaryVertexHighPurBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('simpleSecondaryVertex3TrkComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("secondaryVertexNegativeTagInfos"))
)
# foo bar baz
# oxdIM0AJFhM15
# O7AhrBqNcf2kk
