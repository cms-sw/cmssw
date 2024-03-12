import FWCore.ParameterSet.Config as cms

trackCountingHighEffBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('trackCounting3D2ndComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)


# foo bar baz
# MXTCq8zMjEjXj
