import FWCore.ParameterSet.Config as cms

doubleSecondaryVertexHighEffBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('doubleVertex2TrkComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("inclusiveSecondaryVertexFinderFilteredTagInfos"))
)
# foo bar baz
# kqh0ryvUumGKu
# xGOM5Gt9cVg94
