import FWCore.ParameterSet.Config as cms

jetProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("impactParameterTagInfos"),
    jetTagComputer = cms.string('jetProbability')
)


