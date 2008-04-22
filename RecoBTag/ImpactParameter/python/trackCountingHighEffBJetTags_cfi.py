import FWCore.ParameterSet.Config as cms

trackCountingHighEffBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('trackCounting3D2nd'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)


