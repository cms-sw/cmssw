import FWCore.ParameterSet.Config as cms

simpleSecondaryVertexNegativeHighEffBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('simpleSecondaryVertex2Trk'),
	tagInfos = cms.VInputTag(cms.InputTag("secondaryVertexNegativeTagInfos"))
)
