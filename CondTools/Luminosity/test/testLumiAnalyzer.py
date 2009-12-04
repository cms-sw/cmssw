import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:offlinelumi.db'
process.CondDBCommon.DBParameters.authenticationPath = '.'

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
   process.CondDBCommon,
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   timetype = cms.untracked.string('lumiid'),
   logconnect = cms.untracked.string('sqlite_file:log.db'),
   toGet = cms.VPSet(cms.PSet(
       record = cms.string('LumiSectionDataRcd'),
       tag = cms.string('testlumimixed')
    ))
)

process.maxLuminosityBlocks=cms.untracked.PSet(
   input=cms.untracked.int32(25)
)

process.source = cms.Source("EmptySource",
   numberEventsInRun = cms.untracked.uint32(25),
   firstRun = cms.untracked.uint32(121998),
   lastRun = cms.untracked.uint32(121998),
   numberEventsInLuminosityBlock = cms.untracked.uint32(1),
   firstLuminosityBlock = cms.untracked.uint32(1)
)


#process.source = cms.Source("EmptyIOVSource",
#   timetype = cms.string('lumiid'),
#   firstValue = cms.uint64(4294967297),
#   lastValue = cms.uint64(4294967304),
#   interval = cms.uint64(1)
#)

process.lumianalyzer = cms.EDAnalyzer("LumiDataAnalyzer")

process.p = cms.Path(process.lumianalyzer)
