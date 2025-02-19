import FWCore.ParameterSet.Config as cms

lheZdecayFilter = cms.EDFilter("LHEDYdecayFilter",
    src = cms.InputTag("source"),
    leptonID =cms.int32(11),
    verbose = cms.untracked.bool(True)                                
)                                
