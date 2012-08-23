import FWCore.ParameterSet.Config as cms


simpleInclusiveSecondaryVertexHighEffBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('simpleSecondaryVertex2Trk'),
	tagInfos = cms.VInputTag(cms.InputTag("inclusiveSecondaryVertexFinderTagInfosFiltered"))
)

simpleInclusiveSecondaryVertexHighPurBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('simpleSecondaryVertex3Trk'),
	tagInfos = cms.VInputTag(cms.InputTag("inclusiveSecondaryVertexFinderTagInfosFiltered"))
)
