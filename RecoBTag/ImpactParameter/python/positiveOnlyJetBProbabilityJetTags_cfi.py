import FWCore.ParameterSet.Config as cms

positiveOnlyJetBProbabilityJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveOnlyJetBProbability'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)


