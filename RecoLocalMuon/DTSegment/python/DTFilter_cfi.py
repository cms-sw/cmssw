import FWCore.ParameterSet.Config as cms

DTFilter = cms.EDFilter("DTFilter",
    run = cms.int32(30625),
    useLTC = cms.bool(False),
    LTC_DT = cms.bool(True),
    doRunEvFiltering = cms.bool(False),
    LTC_RPC = cms.bool(True),
    debug = cms.untracked.bool(False),
    LTC_CSC = cms.bool(True),
    event = cms.int32(15095)
)



