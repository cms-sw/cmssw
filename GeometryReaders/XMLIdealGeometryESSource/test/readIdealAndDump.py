import FWCore.ParameterSet.Config as cms

process = cms.Process("CompareGeometryTest")
process.load('Configuration.Geometry.GeometryExtended_cff')

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.myprint = cms.OutputModule("AsciiOutputModule")

process.pAStd = cms.EDAnalyzer("PerfectGeometryAnalyzer"
                               ,dumpPosInfo = cms.untracked.bool(True)
                               ,label = cms.untracked.string("")
                               ,isMagField = cms.untracked.bool(False)
                               ,dumpSpecs = cms.untracked.bool(True)
                               ,dumpGeoHistory = cms.untracked.bool(True)
                               ,outFileName = cms.untracked.string("STD")
                               ,numNodesToDump = cms.untracked.uint32(0)
                               ,fromDB = cms.untracked.bool(False)
                               ,ddRootNodeName = cms.untracked.string("cms:OCMS")
                               )

process.BigXMLWriter = cms.EDAnalyzer("OutputDDToDDL",
                              rotNumSeed = cms.int32(0),
                              fileName = cms.untracked.string("fred.xml")
                              )

process.MessageLogger = cms.Service("MessageLogger",
                                    readIdealerrors = cms.untracked.PSet( threshold = cms.untracked.string('ERROR'),
                                                                          extension = cms.untracked.string('.out')
                                                                          ),
                                    readIdealdebug = cms.untracked.PSet( INFO = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
                                                                         extension = cms.untracked.string('.out'),
                                                                         noLineBreaks = cms.untracked.bool(True),
                                                                         DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
                                                                         threshold = cms.untracked.string('DEBUG'),
                                                                         ),
                                    # For LogDebug/LogTrace output...
                                    debugModules = cms.untracked.vstring('*'),
                                    categories = cms.untracked.vstring('*'),
                                    destinations = cms.untracked.vstring('readIdealerrors','readIdealdebug')
                                    )

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.pAStd)

process.e1 = cms.EndPath(process.myprint)
