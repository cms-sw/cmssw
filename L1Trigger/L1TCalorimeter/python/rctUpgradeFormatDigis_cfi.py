import FWCore.ParameterSet.Config as cms

rctUpgradeFormatDigis = cms.EDProducer(
    "L1TCaloRCTToUpgradeConverter",
    regionTag = cms.InputTag("rctDigis"),
    emTag = cms.InputTag("rctDigis")
)


# foo bar baz
# au2OM4gJKWhxe
# rRILaLSS0PeGX
