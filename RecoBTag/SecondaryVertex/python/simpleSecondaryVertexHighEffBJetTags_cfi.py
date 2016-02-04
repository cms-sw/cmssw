import FWCore.ParameterSet.Config as cms

simpleSecondaryVertexHighEffBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('simpleSecondaryVertex2Trk'),
	tagInfos = cms.VInputTag(cms.InputTag("secondaryVertexTagInfos"))
)
