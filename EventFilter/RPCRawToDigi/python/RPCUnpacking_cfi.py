import FWCore.ParameterSet.Config as cms

rpcunpacker = cms.EDFilter("RPCUnpackingModule",
    PrintOut = cms.untracked.bool(False),
    runDQM = cms.untracked.bool(False),
    PrintHexDump = cms.untracked.bool(False)
)


