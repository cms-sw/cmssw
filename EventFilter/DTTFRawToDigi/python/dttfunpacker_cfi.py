import FWCore.ParameterSet.Config as cms

dttfunpacker = cms.EDProducer("DTTFFEDReader",
    DTTF_FED_Source = cms.InputTag("source"),
    verbose = cms.untracked.bool(False)
)


