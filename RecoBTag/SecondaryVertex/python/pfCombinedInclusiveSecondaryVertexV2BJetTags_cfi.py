import FWCore.ParameterSet.Config as cms

pfCombinedInclusiveSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
        jetTagComputer = cms.string('candidateCombinedSecondaryVertexV2Computer'),
        tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
                                 cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfos"))
)
