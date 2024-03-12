import FWCore.ParameterSet.Config as cms

process = cms.Process("DTDBDUMP")

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:testconf.db'

process.source = cms.Source("EmptyIOVSource",
    timetype   = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue  = cms.uint64(1),
    interval   = cms.uint64(1)
)

process.essource = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    DumpStat=cms.untracked.bool(True),                           
    toGet = cms.VPSet(
    cms.PSet(
    record = cms.string('DTKeyedConfigListRcd'),
    tag = cms.string('DT_keyedConfListIOV_V01')
    ),
    cms.PSet(
    record = cms.string('DTKeyedConfigContainerRcd'),
    tag = cms.string('DT_keyedConfBricks_V01')
    )
    )
)

process.conf_dump = cms.EDAnalyzer("DTKeyedConfigDBDump")

process.p = cms.Path(process.conf_dump)

# foo bar baz
# sZIH4fahg5T0I
# ONFw8epQy1C2c
