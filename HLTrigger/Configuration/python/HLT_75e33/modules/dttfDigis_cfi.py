import FWCore.ParameterSet.Config as cms

dttfDigis = cms.EDProducer("DTTFFEDReader",
    DTTF_FED_Source = cms.InputTag("rawDataCollector"),
    verbose = cms.untracked.bool(False)
)
