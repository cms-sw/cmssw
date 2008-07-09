import FWCore.ParameterSet.Config as cms

l1bias = cms.EDFilter("L1EmulBias",
    RPCsource = cms.InputTag("rpcTriggerDigis"),
    LTCsource = cms.InputTag("none"),
    GLTsource = cms.InputTag("gtModule"),
    HTPsource = cms.InputTag("hcalTriggerPrimitiveDigis"),
    CTFsource = cms.InputTag("csctfDigis"),
    DO_SYSTEM = cms.untracked.vuint32(0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0),
    CTPsource = cms.InputTag("cscTriggerPrimitiveDigis"),
    GMTsource = cms.InputTag("gmtDigis"),
    VerboseFlag = cms.untracked.int32(0),
    DTFsource = cms.InputTag("dttfDigis"),
    CTTsource = cms.InputTag("l1CscTfTrackEmulDigis"),
    DTPsource = cms.InputTag("dtTriggerPrimitiveDigis"),
    ETPsource = cms.InputTag("ecalTriggerPrimitiveDigis"),
    GCTsource = cms.InputTag("gctDigis"),
    RCTsource = cms.InputTag("rctDigis")
)


