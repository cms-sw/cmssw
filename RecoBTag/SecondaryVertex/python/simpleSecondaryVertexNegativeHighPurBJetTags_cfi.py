import FWCore.ParameterSet.Config as cms

simpleSecondaryVertexNegativeHighPurBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('simpleSecondaryVertex3Trk'),
	tagInfos = cms.VInputTag(cms.InputTag("secondaryVertexNegativeTagInfos"))
)
