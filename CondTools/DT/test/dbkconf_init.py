import FWCore.ParameterSet.Config as cms

process = cms.Process("DTDBINIT")

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:testconf.db'

process.source = cms.Source("EmptyIOVSource",
    timetype   = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue  = cms.uint64(1),
    interval   = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    logconnect = cms.untracked.string('sqlite_file:log.db'),
    timetype = cms.untracked.string('runnumber'),
    withWrapper = cms.untracked.bool(False),
    outOfOrder = cms.untracked.bool(True),
    toPut = cms.VPSet(
    cms.PSet(
        record = cms.string('DTCCBConfigRcd'),
        tag = cms.string('conf_test'),
        timetype = cms.untracked.string('runnumber')
    ),
    cms.PSet(
        record = cms.string('keyedConfBricks'),
        tag = cms.string('keyedConfBricks_V01'),
        timetype = cms.untracked.string('hash'),
        withWrapper = cms.untracked.bool(True),
        outOfOrder = cms.untracked.bool(True)
    ),
    cms.PSet(
        record = cms.string('keyedConfListIOV'),
        tag = cms.string('keyedConfListIOV_V01'),
        timetype = cms.untracked.string('runnumber'),
        withWrapper = cms.untracked.bool(True),
        outOfOrder = cms.untracked.bool(False)
    )
    )
)

process.conf_init = cms.EDAnalyzer("DTKeyedConfigDBInit",
    container = cms.string('keyedConfBricks'),
    iov       = cms.string('keyedConfListIOV')
)

process.p = cms.Path(process.conf_init)

