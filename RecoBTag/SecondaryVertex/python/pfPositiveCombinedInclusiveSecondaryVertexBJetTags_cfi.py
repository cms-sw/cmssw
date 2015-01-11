import FWCore.ParameterSet.Config as cms

pfPositiveCombinedInclusiveSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
        jetTagComputer = cms.string('candidatePositiveCombinedSecondaryVertexComputer'),
        tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
                                 cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfos"))
)
