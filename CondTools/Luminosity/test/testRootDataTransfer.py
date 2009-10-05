import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:testroot.db'

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    timetype = cms.untracked.string('lumiid'),
    logconnect = cms.untracked.string('sqlite_file:log.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('LumiSectionData'),
        tag = cms.string('testlumiroot')
    ))
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.lumidatatransfer = cms.EDAnalyzer("LumiSectionDataPopCon",
    Source = cms.PSet(
      lumiRetrieverName = cms.string('rootsource'),
      lumiFileName = cms.string('rfio:/castor/cern.ch/cms/store/lumi/200909/CMS_LUMI_RAW_20090930_001119981_0001_1.root')
    ),
    SinceAppendMode = cms.bool(True),
    name = cms.untracked.string('LumiSectionData'),
    record = cms.string('LumiSectionData'),                     
    loggingOn = cms.untracked.bool(False),
    debug = cms.bool(False)
)

process.p = cms.Path(process.lumidatatransfer)
