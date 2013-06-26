import FWCore.ParameterSet.Config as cms

simpleSecondaryVertexNegativeBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('simpleSecondaryVertex'),
	tagInfos = cms.VInputTag(cms.InputTag("secondaryVertexNegativeTagInfos"))
)
