import FWCore.ParameterSet.Config as cms

CSCTightHaloTrkMuUnvetoFilter = cms.EDFilter(
  "CSCTightHaloTrkMuUnvetoFilter",
  taggingMode = cms.bool(False),
)
