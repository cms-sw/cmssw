import FWCore.ParameterSet.Config as cms

combinedInclusiveSecondaryVertexNegativeBJetTags = cms.EDProducer("JetTagProducer",
        jetTagComputer = cms.string('combinedSecondaryVertexNegative'),
        tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"),
                                 cms.InputTag("inclusiveSecondaryVertexFinderNegativeTagInfos"))
)

combinedInclusiveSecondaryVertexV2NegativeBJetTags = cms.EDProducer("JetTagProducer",
        jetTagComputer = cms.string('combinedSecondaryVertexV2Negative'),
        tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"),
                                 cms.InputTag("inclusiveSecondaryVertexFinderNegativeTagInfos"))
)
