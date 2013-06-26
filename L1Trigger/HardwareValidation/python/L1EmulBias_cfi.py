import FWCore.ParameterSet.Config as cms

l1bias = cms.EDProducer("L1EmulBias",
    ETPsource = cms.InputTag("ecalTriggerPrimitiveDigis"),
    HTPsource = cms.InputTag("hcalTriggerPrimitiveDigis"),
    RCTsource = cms.InputTag("rctDigis"),
    GCTsource = cms.InputTag("gctDigis"),
    DTPsource = cms.InputTag("dtTriggerPrimitiveDigis"),
    DTFsource = cms.InputTag("dttfDigis"),
    CTPsource = cms.InputTag("cscTriggerPrimitiveDigis"),
    CTTsource = cms.InputTag("l1CscTfTrackEmulDigis"),
    CTFsource = cms.InputTag("csctfDigis"),
    RPCsource = cms.InputTag("rpcTriggerDigis"),
    GMTsource = cms.InputTag("gmtDigis"),
    LTCsource = cms.InputTag("none"),
    GLTsource = cms.InputTag("gtModule"),
    DO_SYSTEM = cms.untracked.vuint32(
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    # ETP,HTP,RCT,GCT,DTP,DTF,CTP,CTF,RPC,LTC,GMT,GT
    ),
    VerboseFlag = cms.untracked.int32(0)
)


