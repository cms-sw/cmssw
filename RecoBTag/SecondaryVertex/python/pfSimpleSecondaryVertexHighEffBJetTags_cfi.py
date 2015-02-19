import FWCore.ParameterSet.Config as cms

pfSimpleSecondaryVertexHighEffBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidateSimpleSecondaryVertex2TrkComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("pfSecondaryVertexTagInfos"))
)
