import FWCore.ParameterSet.Config as cms

simpleSecondaryVertexHighEffBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('simpleSecondaryVertex2TrkComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("secondaryVertexTagInfos"))
)
# foo bar baz
# OXRL5WktZkKIm
# i3iOV1D7EDk9Y
