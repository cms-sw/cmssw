import FWCore.ParameterSet.Config as cms

gmtDigis = cms.EDFilter("L1MuGlobalMuonTrigger",
    BX_min_readout = cms.int32(-2),
    BX_min = cms.int32(-4),
    CSCCandidates = cms.InputTag("csctfDigis","CSC"),
    BX_max = cms.int32(4),
    BX_max_readout = cms.int32(2),
    Debug = cms.untracked.int32(0),
    RPCbCandidates = cms.InputTag("rpcTriggerDigis","RPCb"),
    DTCandidates = cms.InputTag("dttfDigis","DT"),
    WriteLUTsAndRegs = cms.untracked.bool(False),
    RPCfCandidates = cms.InputTag("rpcTriggerDigis","RPCf"),
    MipIsoData = cms.InputTag("rctDigis")
)


