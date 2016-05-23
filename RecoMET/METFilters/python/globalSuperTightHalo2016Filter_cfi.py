import FWCore.ParameterSet.Config as cms

globalSuperTightHalo2016Filter = cms.EDFilter(
  "GlobalSuperTightHalo2016Filter",
  taggingMode = cms.bool(False)
)
