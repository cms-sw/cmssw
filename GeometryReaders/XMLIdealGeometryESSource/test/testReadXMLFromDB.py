import FWCore.ParameterSet.Config as cms

process = cms.Process("DBGeometryTest")
#process.load("DetectorDescription.OfflineDBLoader.test.cmsIdealGeometryForWrite_cfi")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )
process.source = cms.Source("EmptySource")

process.myprint = cms.OutputModule("AsciiOutputModule")

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                         process.CondDBSetup,
                                         loadAll = cms.bool(True),
                                         loadBlobStreamer = cms.bool(True),
                                         BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                         toGet = cms.VPSet(cms.PSet(
                                            record = cms.string('GeometryFileRcd'),
                                            tag = cms.string('XMLFILE_TEST_01')
                                         )),
                                         DBParameters = cms.PSet(
                                            messageLevel = cms.untracked.int32(9),
                                            authenticationPath = cms.untracked.string('.')
                                         ),
                                         catalog = cms.untracked.string('file:PoolFileCatalog.xml'),
                                         timetype = cms.string('runnumber'),
                                         connect = cms.string('sqlite_file:myfile.db')
                                      )


process.prod = cms.EDAnalyzer("TestGeometryAnalyzer",
                                  ddRootNodeName = cms.string("cms:OCMS"),
                                  dumpPosInfo = cms.untracked.bool(True),
                                  dumpSpecs = cms.untracked.bool(True),
                                  dumpGeoHistory = cms.untracked.bool(True)
                              )



process.MessageLogger = cms.Service("MessageLogger",
    errors = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR'),
        extension = cms.untracked.string('.out')
    ),
    # No constraint on log content...equivalent to threshold INFO
    # 0 means none, -1 means all (?)
    log = cms.untracked.PSet(
        extension = cms.untracked.string('.out')
    ),
    debug = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        extension = cms.untracked.string('.out'),

        noLineBreaks = cms.untracked.bool(True),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        threshold = cms.untracked.string('DEBUG'),
    ),
    # For LogDebug/LogTrace output...
    debugModules = cms.untracked.vstring('*'),
    categories = cms.untracked.vstring('*'),
    destinations = cms.untracked.vstring('log', 
        'errors', 
        'debug')
)


process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.prod)
process.e1 = cms.EndPath(process.myprint)
