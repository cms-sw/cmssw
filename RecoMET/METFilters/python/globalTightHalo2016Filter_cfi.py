import FWCore.ParameterSet.Config as cms

globalTightHalo2016Filter = cms.EDFilter(
  "GlobalTightHalo2016Filter",
  taggingMode = cms.bool(False)
)
