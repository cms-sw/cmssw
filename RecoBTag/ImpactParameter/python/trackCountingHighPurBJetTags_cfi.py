import FWCore.ParameterSet.Config as cms

trackCountingHighPurBJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("impactParameterTagInfos"),
    jetTagComputer = cms.string('trackCounting3D3rd')
)


