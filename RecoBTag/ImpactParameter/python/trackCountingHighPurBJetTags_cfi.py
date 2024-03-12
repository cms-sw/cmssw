import FWCore.ParameterSet.Config as cms

trackCountingHighPurBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('trackCounting3D3rdComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)


# foo bar baz
# cb9qgkLgY7QGc
# pVqj1scpKSbeC
