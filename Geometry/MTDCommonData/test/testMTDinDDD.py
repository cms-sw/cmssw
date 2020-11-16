import FWCore.ParameterSet.Config as cms

process = cms.Process("CompareGeometryTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.MessageLogger = cms.Service(
    "MessageLogger",
    statistics = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('MTDUnitTest',
                                       'TestMTDIdealGeometry',
                                       'TestMTDPath',
                                       'TestMTDNumbering',
                                       'TestMTDPosition'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            ),
        FWKINFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            ),
        TestMTDIdealGeometry = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            ),
        TestMTDPath = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            ),
        TestMTDNumbering = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            ),
        TestMTDPosition = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            ),
        noLineBreaks = cms.untracked.bool(True)
        ),
    mtdCommonDataDDD = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            ),
        FWKINFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            ),
        noLineBreaks = cms.untracked.bool(True),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            ),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            ),
        ERROR = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            ),
        threshold = cms.untracked.string('INFO'),
        MTDUnitTest = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            ),
        ),
    destinations = cms.untracked.vstring('cout',
                                         'mtdCommonDataDDD')
)

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

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.testBTL+process.testETL)
