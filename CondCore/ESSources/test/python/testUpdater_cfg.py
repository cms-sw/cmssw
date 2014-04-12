import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:Ints.db'

process.eff = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    RefreshEachRun=cms.untracked.bool(True),
    DumpStat=cms.untracked.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('OneIntRcd'),
        tag = cms.string('OneInt')
    ))
)

#process.source = cms.Source("EmptyIOVSource",
#    firstValue = cms.uint64(310),
#    lastValue = cms.uint64(1000),
#    timetype = cms.string('runnumber'),
#    interval = cms.uint64(100)
#)

process.source = cms.Source("EmptySource",
  firstLuminosityBlock = cms.untracked.uint32(1),
  numberEventsInLuminosityBlock = cms.untracked.uint32(3),
  firstEvent = cms.untracked.uint32(1),
  firstRun = cms.untracked.uint32(10),
  numberEventsInRun = cms.untracked.uint32(12)
)

process.maxEvents = cms.untracked.PSet(
 input = cms.untracked.int32(400)
)

process.prod = cms.EDAnalyzer("TestUpdater")

process.p = cms.Path(process.prod)

