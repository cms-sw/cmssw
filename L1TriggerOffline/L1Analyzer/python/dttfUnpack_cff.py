import FWCore.ParameterSet.Config as cms
l1dttfunpack = cms.EDProducer("DTTFFEDReader",
     DTTF_FED_Source = cms.InputTag("source")
)
