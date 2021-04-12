import FWCore.ParameterSet.Config as cms

simGmtDigis = cms.EDProducer("L1MuGlobalMuonTrigger",
    BX_max = cms.int32(4),
    BX_max_readout = cms.int32(2),
    BX_min = cms.int32(-4),
    BX_min_readout = cms.int32(-2),
    CSCCandidates = cms.InputTag("simCsctfDigis","CSC"),
    DTCandidates = cms.InputTag("simDttfDigis","DT"),
    Debug = cms.untracked.int32(0),
    MipIsoData = cms.InputTag("simRctDigis"),
    RPCbCandidates = cms.InputTag("simRpcTriggerDigis","RPCb"),
    RPCfCandidates = cms.InputTag("simRpcTriggerDigis","RPCf"),
    SendMipIso = cms.untracked.bool(False),
    WriteLUTsAndRegs = cms.untracked.bool(False)
)
