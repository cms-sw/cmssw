import FWCore.ParameterSet.Config as cms

negativeCombinedInclusiveSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
        jetTagComputer = cms.string('negativeCombinedSecondaryVertexComputer'),
        tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"),
                                 cms.InputTag("inclusiveSecondaryVertexFinderNegativeTagInfos"))
)
