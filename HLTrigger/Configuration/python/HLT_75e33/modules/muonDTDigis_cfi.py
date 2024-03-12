import FWCore.ParameterSet.Config as cms

muonDTDigis = cms.EDProducer("DTuROSRawToDigi",
    debug = cms.untracked.bool(False),
    inputLabel = cms.InputTag("rawDataCollector")
)
# foo bar baz
# 1rjGFYsPeC9aV
# ZotsdgbitRPXz
