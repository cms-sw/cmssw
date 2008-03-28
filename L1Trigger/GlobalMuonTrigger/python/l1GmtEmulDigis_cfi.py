import FWCore.ParameterSet.Config as cms

l1GmtEmulDigis = cms.EDFilter("L1MuGlobalMuonTrigger",
    BX_min_readout = cms.int32(-2),
    BX_min = cms.int32(-4),
    CSCCandidates = cms.InputTag("l1CscTfEmulDigis","CSC"),
    BX_max = cms.int32(4),
    BX_max_readout = cms.int32(2),
    Debug = cms.untracked.int32(0),
    RPCbCandidates = cms.InputTag("l1RpcEmulDigis","RPCb"),
    DTCandidates = cms.InputTag("l1DttfEmulDigis","DT"),
    WriteLUTsAndRegs = cms.untracked.bool(False),
    RPCfCandidates = cms.InputTag("l1RpcEmulDigis","RPCf"),
    MipIsoData = cms.InputTag("L1RCTRegionSumsEmCands")
)


