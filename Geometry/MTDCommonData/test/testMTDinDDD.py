import FWCore.ParameterSet.Config as cms

process = cms.Process("CompareGeometryTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        FWKINFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        TestMTDIdealGeometry = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        TestMTDNumbering = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        TestMTDPath = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        TestMTDPosition = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        enable = cms.untracked.bool(True),
        enableStatistics = cms.untracked.bool(True),
        noLineBreaks = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    ),
    files = cms.untracked.PSet(
        mtdCommonDataDDD = cms.untracked.PSet(
            DEBUG = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            ERROR = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            FWKINFO = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            INFO = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            MTDUnitTest = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            WARNING = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            noLineBreaks = cms.untracked.bool(True),
            threshold = cms.untracked.string('INFO')
        )
    )
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
