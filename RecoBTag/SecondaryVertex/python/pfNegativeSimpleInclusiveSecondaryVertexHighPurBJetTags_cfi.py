import FWCore.ParameterSet.Config as cms

pfNegativeSimpleInclusiveSecondaryVertexHighPurBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidateSimpleSecondaryVertex3TrkComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("pfInclusiveSecondaryVertexFinderNegativeTagInfos"))
)
