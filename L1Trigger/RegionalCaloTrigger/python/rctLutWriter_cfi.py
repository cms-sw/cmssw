import FWCore.ParameterSet.Config as cms

rctLutWriter = cms.EDAnalyzer("L1RCTLutWriter",
    useDebugTpgScales = cms.bool(False),
    key = cms.string('dummy')
)



