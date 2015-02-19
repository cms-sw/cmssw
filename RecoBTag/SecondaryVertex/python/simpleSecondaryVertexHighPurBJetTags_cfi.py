import FWCore.ParameterSet.Config as cms

simpleSecondaryVertexHighPurBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('simpleSecondaryVertex3TrkComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("secondaryVertexTagInfos"))
)
