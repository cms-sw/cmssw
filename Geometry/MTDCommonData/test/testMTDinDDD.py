import FWCore.ParameterSet.Config as cms

process = cms.Process("CompareGeometryTest")
process.load('Configuration.Geometry.GeometryExtended2026D50_cff')

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

process.testBTL = cms.EDAnalyzer("TestMTDNumbering",
                               label = cms.untracked.string(''),
                               outFileName = cms.untracked.string('BTL'),
                               ddTopNodeName = cms.untracked.string('BarrelTimingLayer'),
                               theLayout = cms.untracked.uint32(4)
                               )

process.testETL = cms.EDAnalyzer("TestMTDNumbering",
                               label = cms.untracked.string(''),
                               outFileName = cms.untracked.string('ETL'),
                               ddTopNodeName = cms.untracked.string('EndcapTimingLayer')
                               )


process.testBTLpos = cms.EDAnalyzer("TestMTDPosition",
                               label = cms.untracked.string(''),
                               outFileName = cms.untracked.string('BTLpos'),
                               ddTopNodeName = cms.untracked.string('BarrelTimingLayer')
                               )

process.testETLpos = cms.EDAnalyzer("TestMTDPosition",
                               label = cms.untracked.string(''),
                               outFileName = cms.untracked.string('ETLpos'),
                               ddTopNodeName = cms.untracked.string('EndcapTimingLayer')
                               )

process.MessageLogger = cms.Service("MessageLogger",
                                    cout = cms.untracked.PSet( INFO = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
                                                               noLineBreaks = cms.untracked.bool(True),
                                                               threshold = cms.untracked.string('INFO'),
                                                               ),
                                    # For LogDebug/LogTrace output...
                                    categories = cms.untracked.vstring('TestMTDNumbering','MTDGeom','TestMTDPosition'),
                                    destinations = cms.untracked.vstring('cout')
                                    )

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.testBTL+process.testETL+process.testBTLpos+process.testETLpos)

process.e1 = cms.EndPath(process.myprint)
