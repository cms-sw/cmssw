import FWCore.ParameterSet.Config as cms

combinedInclusiveSecondaryVertexPositiveBJetTags = cms.EDProducer("JetTagProducer",
        jetTagComputer = cms.string('combinedSecondaryVertexPositive'),
        tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"),
                                 cms.InputTag("inclusiveSecondaryVertexFinderTagInfos"))
)
