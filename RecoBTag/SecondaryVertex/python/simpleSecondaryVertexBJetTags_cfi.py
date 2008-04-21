import FWCore.ParameterSet.Config as cms

simpleSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
	tagInfo = cms.InputTag("secondaryVertexTagInfos"),
	jetTagComputer = cms.string('simpleSecondaryVertex')
)


