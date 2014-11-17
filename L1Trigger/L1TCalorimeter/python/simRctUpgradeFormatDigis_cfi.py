import FWCore.ParameterSet.Config as cms

simRctUpgradeFormatDigis = cms.EDProducer(
    "L1TCaloRCTToUpgradeConverter",
    regionTag = cms.InputTag("simRctDigis"),
    emTag = cms.InputTag("simRctDigis")
)


