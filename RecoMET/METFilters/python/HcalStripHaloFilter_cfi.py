import FWCore.ParameterSet.Config as cms

HcalStripHaloFilter = cms.EDFilter(
  "HcalStripHaloFilter",
  taggingMode = cms.bool(False),
  maxWeightedStripLength = cms.int32(7),
  maxEnergyRatio = cms.double(0.15),
  minHadEt = cms.double(100.0)
)
# foo bar baz
# 4eZwrwcVZVdNI
# CbnomrRj9VL0x
