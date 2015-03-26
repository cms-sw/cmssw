import FWCore.ParameterSet.Config as cms

pfNegativeSimpleSecondaryVertexHighPurBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidateSimpleSecondaryVertex3TrkComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("pfSecondaryVertexNegativeTagInfos"))
)
