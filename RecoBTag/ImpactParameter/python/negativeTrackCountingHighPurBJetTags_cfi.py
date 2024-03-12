import FWCore.ParameterSet.Config as cms

negativeTrackCountingHighPurBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeTrackCounting3D3rdComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)
# foo bar baz
# pU82tSCt1p1V0
