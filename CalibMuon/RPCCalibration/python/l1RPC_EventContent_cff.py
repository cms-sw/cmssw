import FWCore.ParameterSet.Config as cms

l1RPC_EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
        'keep RPCDetIdRPCDigiMuonDigiCollection_*_*_*', 
        'keep L1MuGMTCands_*_*_*', 
        'keep L1MuGMTReadoutCollection_*_*_*')
)
l1RPCEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('l1RPCHLTPath')
    )
)

