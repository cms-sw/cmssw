import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:offlinelumi.db'

process.MessageLogger = cms.Service("MessageLogger",
   suppressInfo = cms.untracked.vstring(),
   destinations = cms.untracked.vstring('joboutput'),
   categories = cms.untracked.vstring('LumiReport'),
   joboutput = cms.untracked.PSet(
     threshold = cms.untracked.string('INFO'),
     noLineBreaks = cms.untracked.bool(True),
     noTimeStamps = cms.untracked.bool(True),
     INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
     LumiReport = cms.untracked.PSet( limit = cms.untracked.int32(10000000) )
   )
)

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
      lumiFileName = cms.string('/afs/cern.ch/user/x/xiezhen/w1/CMS_LUMI_RAW_20091110_000120020_0001_1.root'),
      allowForceFirstSince = cms.bool(True)
    ),                                          
    SinceAppendMode = cms.bool(True),
    name = cms.untracked.string('LumiSectionData'),
    record = cms.string('LumiSectionData'),                     
    loggingOn = cms.untracked.bool(True),
    debug = cms.bool(False)
)

process.p = cms.Path(process.lumidatatransfer)
