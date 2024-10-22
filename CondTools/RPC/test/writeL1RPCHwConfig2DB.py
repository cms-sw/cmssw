import FWCore.ParameterSet.Config as cms

process = cms.Process("Write2DB")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:L1RPCHwConfig.db'

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True)
    )
)

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(12505),
    lastValue = cms.uint64(12505),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    logconnect = cms.untracked.string('sqlite_file:L1RPCHwConfig_log.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('L1RPCHwConfigRcd'),
        tag = cms.string('L1RPCHwConfig_v1')
    ))
)

process.WriteInDB = cms.EDFilter("L1RPCHwConfigDBWriter",
    SinceAppendMode = cms.bool(True),
    record = cms.string('L1RPCHwConfigRcd'),
    loggingOn = cms.untracked.bool(True),
    Source = cms.PSet(
        FirstBX = cms.untracked.int32(0),
        LastBX = cms.untracked.int32(0),
#        DisabledCrates = cms.untracked.vint32(0,1,2,3,4,5),
        DisabledCrates = cms.untracked.vint32(),
        DisabledTowers = cms.untracked.vint32(),
        WriteDummy = cms.untracked.int32(1),
        Validate = cms.untracked.int32(1),
        OnlineAuthPath = cms.untracked.string('.'),
        OnlineConn = cms.untracked.string('oracle://cms_omds_lb/CMS_RPC_CONF')
    )
)

process.p = cms.Path(process.WriteInDB)

