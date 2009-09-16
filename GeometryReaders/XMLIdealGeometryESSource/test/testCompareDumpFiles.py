import FWCore.ParameterSet.Config as cms

process = cms.Process("CompareGeometryTest")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

#process.demo = cms.EDAnalyzer("PrintEventSetupContent")

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.myprint = cms.OutputModule("AsciiOutputModule")

process.comparedddump = cms.EDAnalyzer("TestCompareDDDumpFiles"
                                       ,dumpFile1 = cms.string("workarea/xml/dumpSTD")
                                       ,dumpFile2 = cms.string("workarea/db/dumpBDB")
                                       ,tolerance = cms.untracked.double(0.0004)
                                       )

process.MessageLogger = cms.Service("MessageLogger",
                                    compDDdumperrors = cms.untracked.PSet( threshold = cms.untracked.string('ERROR'),
                                                                          extension = cms.untracked.string('.out')
                                                                          ),
                                    compDDdumpdebug = cms.untracked.PSet( INFO = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
                                                                         extension = cms.untracked.string('.out'),
                                                                         noLineBreaks = cms.untracked.bool(True),
                                                                         DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
                                                                         threshold = cms.untracked.string('DEBUG'),
                                                                         ),
                                    # For LogDebug/LogTrace output...
                                    debugModules = cms.untracked.vstring('*'),
                                    categories = cms.untracked.vstring('*'),
                                    destinations = cms.untracked.vstring('compDDdumperrors','compDDdumpdebug')
                                    )

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.comparedddump)
process.e1 = cms.EndPath(process.myprint)
