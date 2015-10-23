import FWCore.ParameterSet.Config as cms

HcalStripHaloFilter = cms.EDFilter(
  "HcalStripHaloFilter",
  taggingMode = cms.bool(False),
  maxWeightedStripLength = cms.int32(9) # values higher than this are rejected
)
