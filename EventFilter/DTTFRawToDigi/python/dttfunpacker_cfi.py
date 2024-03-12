import FWCore.ParameterSet.Config as cms

dttfunpacker = cms.EDProducer("DTTFFEDReader",
    DTTF_FED_Source = cms.InputTag("rawDataCollector"),
    verbose = cms.untracked.bool(False)
)


# foo bar baz
# 9AAxEhcbuszvV
