import FWCore.ParameterSet.Config as cms

pfNegativeSimpleInclusiveSecondaryVertexHighEffBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('candidateSimpleSecondaryVertex2TrkComputer'),
	tagInfos = cms.VInputTag(cms.InputTag("pfInclusiveSecondaryVertexFinderNegativeTagInfos"))
)
