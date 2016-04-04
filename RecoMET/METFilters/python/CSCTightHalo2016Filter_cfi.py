import FWCore.ParameterSet.Config as cms

CSCTightHalo2016Filter = cms.EDFilter(
  "CSCTightHalo2016Filter",
  taggingMode = cms.bool(False)
)
