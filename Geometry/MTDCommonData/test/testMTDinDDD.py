import FWCore.ParameterSet.Config as cms

process = cms.Process("CompareGeometryTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.INFO.limit = -1

process.load('Configuration.Geometry.GeometryExtended2026D50_cff')

process.testBTL = cms.EDAnalyzer("TestMTDIdealGeometry",
                               label = cms.untracked.string(''),
                               ddTopNodeName = cms.untracked.string('BarrelTimingLayer'),
                               theLayout = cms.untracked.uint32(4)
                               )

process.testETL = cms.EDAnalyzer("TestMTDIdealGeometry",
                               label = cms.untracked.string(''),
                               ddTopNodeName = cms.untracked.string('EndcapTimingLayer'),
                               theLayout = cms.untracked.uint32(4)
                               )

process.MessageLogger = cms.Service("MessageLogger",
                                    cout = cms.untracked.PSet( INFO = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
                                                               noLineBreaks = cms.untracked.bool(True),
                                                               threshold = cms.untracked.string('INFO'),
                                                               ),
                                    # For LogDebug/LogTrace output...
                                    categories = cms.untracked.vstring('TestMTDIdealGeometry','MTDGeom','TestMTDPosition'),
                                    destinations = cms.untracked.vstring('cout')
                                    )

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.testBTL+process.testETL)
