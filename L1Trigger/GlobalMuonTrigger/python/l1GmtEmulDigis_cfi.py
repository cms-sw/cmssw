import FWCore.ParameterSet.Config as cms

l1GmtEmulDigis = cms.EDProducer("L1MuGlobalMuonTrigger",
    Debug = cms.untracked.int32(0),
    BX_min = cms.int32(-4),
    BX_max = cms.int32(4),
    BX_min_readout = cms.int32(-2),
    BX_max_readout = cms.int32(2),
    DTCandidates = cms.InputTag("l1DttfEmulDigis","DT"),
    RPCbCandidates = cms.InputTag("l1RpcEmulDigis","RPCb"),
    CSCCandidates = cms.InputTag("l1CscTfEmulDigis","CSC"),
    RPCfCandidates = cms.InputTag("l1RpcEmulDigis","RPCf"),
    MipIsoData = cms.InputTag("L1RCTRegionSumsEmCands"),
    WriteLUTsAndRegs = cms.untracked.bool(False),
    SendMipIso = cms.untracked.bool(False)
)


