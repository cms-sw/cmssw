import FWCore.ParameterSet.Config as cms

pfNegativeCombinedInclusiveSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
        jetTagComputer = cms.string('candidateNegativeCombinedSecondaryVertexComputer'),
        tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
                                 cms.InputTag("pfInclusiveSecondaryVertexFinderNegativeTagInfos"))
)
