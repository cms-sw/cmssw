import FWCore.ParameterSet.Config as cms

process = cms.Process("DBTest")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.RPCCabling = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('RPCEMapRcd'),
        tag = cms.string('RPCEMap_v2')
    )),
    connect = cms.string('sqlite_file:RPCEMap.db')
)

process.reader = cms.EDAnalyzer("RPCReadOutMapAnalyzer",
    useNewEMap = cms.untracked.bool(True)
)

process.p1 = cms.Path(process.reader)


