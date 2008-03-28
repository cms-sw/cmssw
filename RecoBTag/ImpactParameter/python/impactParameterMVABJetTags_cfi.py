import FWCore.ParameterSet.Config as cms

impactParameterMVABJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("impactParameterTagInfos"),
    jetTagComputer = cms.string('impactParameterMVAComputer')
)


