import FWCore.ParameterSet.Config as cms

CSCTightHalo2015Filter = cms.EDFilter(
  "CSCTightHalo2015Filter",
  taggingMode = cms.bool(False)
)
