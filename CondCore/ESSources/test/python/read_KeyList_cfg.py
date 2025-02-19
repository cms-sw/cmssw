import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:keys.db'

process.eff = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    DumpStat=cms.untracked.bool(True),                           
    toGet = cms.VPSet(
    cms.PSet(
    record = cms.string('ExDwarfListRcd'),
    tag = cms.string('ConfTest')
    ),
    cms.PSet(
    record = cms.string('ExDwarfRcd'),
    tag = cms.string('KeyTest')
    )
    )
)

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(15),
    lastValue = cms.uint64(50),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(7)
)

process.prod = cms.EDAnalyzer("KeyListAnalyzer")

process.p = cms.Path(process.prod)

