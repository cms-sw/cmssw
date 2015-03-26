import FWCore.ParameterSet.Config as cms

pfSimpleSecondaryVertexHighPurBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidateSimpleSecondaryVertex3TrkComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("pfSecondaryVertexTagInfos"))
)
