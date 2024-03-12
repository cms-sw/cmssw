import FWCore.ParameterSet.Config as cms

pfNegativeCombinedInclusiveSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
        jetTagComputer = cms.string('candidateNegativeCombinedSecondaryVertexV2Computer'),
        tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
                                 cms.InputTag("pfInclusiveSecondaryVertexFinderNegativeTagInfos"))
)
# foo bar baz
# 1afGqUm9l0Ya5
# 56ZXGE25lqmr4
