import FWCore.ParameterSet.Config as cms

negativeSimpleInclusiveSecondaryVertexHighPurBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('simpleSecondaryVertex3TrkComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("inclusiveSecondaryVertexFinderFilteredNegativeTagInfos"))
)
