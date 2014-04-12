mport FWCore.ParameterSet.Config as cms

process = cms.Process("Write2DB")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:RPCEMap.db'

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    logconnect = cms.untracked.string('sqlite_file:RPCEMap_log.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('RPCEMapRcd'),
        tag = cms.string('RPCEMap_v2')
    ))
)

process.WriteInDB = cms.EDFilter("RPCEMapDBWriter",
    SinceAppendMode = cms.bool(True),
    record = cms.string('RPCEMapRcd'),
    loggingOn = cms.untracked.bool(False),
    Source = cms.PSet(
        OnlineAuthPath = cms.untracked.string('.'),
        Validate = cms.untracked.int32(0),
        OnlineConn = cms.untracked.string('oracle://cms_omds_lb/CMS_RPC_CONF')
    )
)

process.p = cms.Path(process.WriteInDB)

