import FWCore.ParameterSet.Config as cms

dttfunpacker = cms.EDFilter("DTTFFEDReader",
    DTTF_FED_Source = cms.InputTag("source")
)


