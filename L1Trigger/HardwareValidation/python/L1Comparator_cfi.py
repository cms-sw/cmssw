import FWCore.ParameterSet.Config as cms

l1compare = cms.EDFilter("L1Comparator",
    ETPsourceData = cms.InputTag("ecalTriggerPrimitiveDigis"),
    ETPsourceEmul = cms.InputTag("valEcalTriggerPrimitiveDigis"),
    HTPsourceData = cms.InputTag("hcalTriggerPrimitiveDigis"),
    HTPsourceEmul = cms.InputTag("valHcalTriggerPrimitiveDigis"),
    RCTsourceData = cms.InputTag("gctDigis"),
    RCTsourceEmul = cms.InputTag("valRctDigis"),
    GCTsourceData = cms.InputTag("gctDigis"),
    GCTsourceEmul = cms.InputTag("valGctDigis"),
    ##csc tp comparison: i)csc-readout vs csc-emulator
    CTPsourceData = cms.InputTag("muonCSCDigis"),
    CTPsourceEmul = cms.InputTag("valCscTriggerPrimitiveDigis"),
    ##csc tp comparison: ii) csctf-readout vs csc-emulator
    #CTPsourceData = cms.InputTag("csctfDigis"),
    #CTPsourceEmul = cms.InputTag("valCscTriggerPrimitiveDigis","MPCSORTED"),
    CTTsourceData = cms.InputTag("csctfDigis"),
    CTTsourceEmul = cms.InputTag("valCsctfTrackDigis"),
    CTFsourceData = cms.InputTag("muonCscMon","CSC"),
    CTFsourceEmul = cms.InputTag("valCsctfDigis","CSC"),
    DTPsourceData = cms.InputTag("dttfDigis"),
    DTPsourceEmul = cms.InputTag("valDtTriggerPrimitiveDigis"),
    DTFsourceData = cms.InputTag("muonDtMon","DT"),
    DTFsourceEmul = cms.InputTag("valDttfDigis"),
    RPCsourceData = cms.InputTag("gtDigis"),
    RPCsourceEmul = cms.InputTag("valRpcTriggerDigis"),
    GMTsourceData = cms.InputTag("gtDigis"),
    GMTsourceEmul = cms.InputTag("valGmtDigis"),
    GLTsourceData = cms.InputTag("gtDigis"),
    GLTsourceEmul = cms.InputTag("valGtDigis"),
    LTCsourceData = cms.InputTag("none"),
    LTCsourceEmul = cms.InputTag("none"),
    FEDsourceData = cms.untracked.InputTag("source"),
    FEDsourceEmul = cms.untracked.InputTag("source"),
    FEDid = cms.untracked.int32(735),
    DumpMode = cms.untracked.int32(0),
    DumpFile = cms.untracked.string('dump.txt'),
    VerboseFlag = cms.untracked.int32(0),
    COMPARE_COLLS = cms.untracked.vuint32(
    1, 1, 1, 1,  1, 1, 1, 1, 1,  0, 1, 1
    # ETP,HTP,RCT,GCT, DTP,DTF,CTP,CTF,RPC, LTC,GMT,GT
    )
)



