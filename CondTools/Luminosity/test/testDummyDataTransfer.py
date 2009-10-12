import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
#process.CondDBCommon.connect = 'sqlite_file:offlinelumi.db'
process.CondDBCommon.connect = 'oracle://cms_orcon_prod/CMS_COND_31X_RUN_INFO'
process.CondDBCommon.DBParameters.authenticationPath = '/nfshome0/xiezhen/conddb'

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    timetype = cms.untracked.string('lumiid'),
    logconnect = cms.untracked.string('sqlite_file:log.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('LumiSectionData'),
        tag = cms.string('lhcscan_1')
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
      lumiRetrieverName = cms.string('dummysource'),
      runNumber = cms.int32(1),
      lumiVersion = cms.string('1'),
      allowForceFirstSince = cms.bool(False)
    ),
    SinceAppendMode = cms.bool(True),
    name = cms.untracked.string('LumiSectionData'),
    record = cms.string('LumiSectionData'),                     
    loggingOn = cms.untracked.bool(True),
    debug = cms.bool(False)
)

process.p = cms.Path(process.lumidatatransfer)
