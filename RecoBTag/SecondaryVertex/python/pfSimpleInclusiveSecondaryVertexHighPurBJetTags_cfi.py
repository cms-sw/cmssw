import FWCore.ParameterSet.Config as cms

pfSimpleInclusiveSecondaryVertexHighPurBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidateSimpleSecondaryVertex3TrkComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfos"))
)
