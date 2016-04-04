import FWCore.ParameterSet.Config as cms

CSCSuperTightHalo2016Filter = cms.EDFilter(
  "CSCSuperTightHalo2016Filter",
  taggingMode = cms.bool(False)
)
