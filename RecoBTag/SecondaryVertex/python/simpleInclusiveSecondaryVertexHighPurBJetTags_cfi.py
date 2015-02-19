import FWCore.ParameterSet.Config as cms

simpleInclusiveSecondaryVertexHighPurBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('simpleSecondaryVertex3TrkComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("inclusiveSecondaryVertexFinderFilteredTagInfos"))
)
