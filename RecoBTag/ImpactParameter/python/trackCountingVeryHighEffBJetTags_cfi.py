import FWCore.ParameterSet.Config as cms

trackCountingVeryHighEffBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('trackCounting3D1st'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)


