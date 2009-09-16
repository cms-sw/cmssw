import FWCore.ParameterSet.Config as cms

process = cms.Process("DBGeometryTest")
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
                                      toGet = cms.VPSet(cms.PSet(record = cms.string('GeometryFileRcd'),
                                                                 tag = cms.string('XMLFILE_Geometry_TagXX')
                                                                 )
                                                        ),
                                      catalog = cms.untracked.string('file:PoolFileCatalog.xml'),
                                      timetype = cms.string('runnumber'),
                                      connect = cms.string('sqlite_file:myfile.db')
                                      )

process.PoolDBESSource.DBParameters = cms.PSet(
    messageLevel = cms.untracked.int32(9),
    authenticationPath = cms.untracked.string('.')
    )

process.pDB = cms.EDAnalyzer("PerfectGeometryAnalyzer"
                               ,dumpPosInfo = cms.untracked.bool(True)
                               ,label = cms.untracked.string("")
                               ,isMagField = cms.untracked.bool(False)
                               ,dumpSpecs = cms.untracked.bool(True)
                               ,dumpGeoHistory = cms.untracked.bool(True)
                               ,outFileName = cms.untracked.string("BDB")
                               ,numNodesToDump = cms.untracked.uint32(0)
                               ,fromDB = cms.untracked.bool(True)
                               ,ddRootNodeName = cms.untracked.string("cms:OCMS")
                               )

process.MessageLogger = cms.Service("MessageLogger",
                                    readDBerrors = cms.untracked.PSet( threshold = cms.untracked.string('ERROR'),
                                                                          extension = cms.untracked.string('.out')
                                                                          ),
                                    readDBdebug = cms.untracked.PSet( INFO = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
                                                                         extension = cms.untracked.string('.out'),
                                                                         noLineBreaks = cms.untracked.bool(True),
                                                                         DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
                                                                         threshold = cms.untracked.string('DEBUG'),
                                                                         ),
                                    # For LogDebug/LogTrace output...
                                    debugModules = cms.untracked.vstring('*'),
                                    categories = cms.untracked.vstring('*'),
                                    destinations = cms.untracked.vstring('readDBerrors','readDBdebug')
                                    )

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.pDB)
process.e1 = cms.EndPath(process.myprint)
