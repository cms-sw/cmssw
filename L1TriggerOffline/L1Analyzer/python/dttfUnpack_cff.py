import FWCore.ParameterSet.Config as cms
l1dttfunpack = cms.EDFilter("DTTFFEDReader",
     DTTF_FED_Source = cms.InputTag("source")
)
