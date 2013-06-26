import FWCore.ParameterSet.Config as cms

CSCTightHaloFilter = cms.EDFilter(
  "CSCTightHaloFilter",
  taggingMode = cms.bool(False),
)
