import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:Ints.db'
# process.CondDBCommon.DBParameters.messageLevel = cms.untracked.int32(3)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
    withWrapper = cms.untracked.bool(False),
    outOfOrder = cms.untracked.bool(True),
    toPut = cms.VPSet(
    cms.PSet(
    record = cms.string('oneInt'),
    tag = cms.string('OneInt'),
    timetype = cms.untracked.string('runnumber'),
    withWrapper = cms.untracked.bool(True),
    outOfOrder = cms.untracked.bool(False)
    )
    )
)

process.mytest = cms.EDAnalyzer("writeInt",
                                Number=cms.int32(10)
                                )

process.p = cms.Path(process.mytest)

