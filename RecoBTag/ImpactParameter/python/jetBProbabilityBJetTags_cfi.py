import FWCore.ParameterSet.Config as cms

jetBProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("impactParameterTagInfos"),
    jetTagComputer = cms.string('jetBProbability')
)


