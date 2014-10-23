import FWCore.ParameterSet.Config as cms

combinedInclusiveSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('combinedSecondaryVertex'),
	tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"),
	                         cms.InputTag("inclusiveSecondaryVertexFinderTagInfos"))
)

combinedInclusiveSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('combinedSecondaryVertexV2'),
	tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"),
	                         cms.InputTag("inclusiveSecondaryVertexFinderTagInfos"))
)

pfCombinedInclusiveSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
        jetTagComputer = cms.string('candidateCombinedSecondaryVertexV2'),
        tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
                                 cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfos"))
)

