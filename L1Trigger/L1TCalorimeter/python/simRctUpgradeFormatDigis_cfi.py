import FWCore.ParameterSet.Config as cms

simRctUpgradeFormatDigis = cms.EDProducer(
    "l1t::L1TCaloRCTToUpgradeConverter",
    regionTag = cms.InputTag("simRctDigis"),
    emTag = cms.InputTag("simRctDigis")
)


