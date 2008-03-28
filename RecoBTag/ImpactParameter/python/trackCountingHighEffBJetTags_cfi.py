import FWCore.ParameterSet.Config as cms

trackCountingHighEffBJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("impactParameterTagInfos"),
    jetTagComputer = cms.string('trackCounting3D2nd')
)


