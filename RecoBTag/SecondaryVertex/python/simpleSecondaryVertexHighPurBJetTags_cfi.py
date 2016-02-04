import FWCore.ParameterSet.Config as cms

simpleSecondaryVertexHighPurBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('simpleSecondaryVertex3Trk'),
	tagInfos = cms.VInputTag(cms.InputTag("secondaryVertexTagInfos"))
)
