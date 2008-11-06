import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.source = cms.Source("EmptyIOVSource",
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(4294967295),
                            lastValue = cms.uint64(4294967295),
                            interval = cms.uint64(1)
                            )

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.orcon = cms.ESSource("PoolDBESSource",
                             process.CondDBCommon,
                             toGet =
                             cms.VPSet(cms.PSet(
    record = cms.string('L1TriggerKeyListRcd'),
    tag = cms.string('L1TriggerKeyList_IDEAL'))
    )
)

process.orcon.connect = cms.string('sqlite_file:l1config.db')

process.demo = cms.EDAnalyzer('L1O2OTestAnalyzer'
)


process.p = cms.Path(process.demo)
