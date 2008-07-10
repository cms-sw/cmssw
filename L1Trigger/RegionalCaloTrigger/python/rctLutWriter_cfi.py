import FWCore.ParameterSet.Config as cms

rctLutWriter = cms.EDFilter("L1RCTLutWriter",
    useDebugTpgScales = cms.bool(False),
    key = cms.string('dummy')
)



