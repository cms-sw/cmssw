import FWCore.ParameterSet.Config as cms

process = cms.Process("PVSSIDReader")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:PVSStoID.db'
process.CondDBCommon.DBParameters.authenticationPath = './'


process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('timestamp'),
    firstValue = cms.uint64(1233913599),
    lastValue = cms.uint64(1233913599),
    interval = cms.uint64(1)
)



process.rn = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    timetype = cms.string('timestamp'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('RPCObPVSSmapRcd'),
        tag = cms.string('Imon_v3')
    ))
)


process.prod = cms.EDAnalyzer("PVSSIDReader")
process.p = cms.Path(process.prod)


