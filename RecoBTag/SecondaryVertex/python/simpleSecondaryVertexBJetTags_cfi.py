import FWCore.ParameterSet.Config as cms

simpleSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('simpleSecondaryVertex'),
	tagInfos = cms.VInputTag(cms.InputTag("secondaryVertexTagInfos"))
)
