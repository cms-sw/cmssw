import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

process = cms.Process("CompareGeometryTest",Phase2C17I13M9)

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = cms.untracked.string('INFO')
process.MessageLogger.cerr.INFO = cms.untracked.PSet(
    limit = cms.untracked.int32(0)
)
process.MessageLogger.cerr.TestMTDIdealGeometry = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)
process.MessageLogger.cerr.TestMTDNumbering = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)
process.MessageLogger.cerr.TestMTDPath = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)
process.MessageLogger.cerr.TestMTDPosition = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)
process.MessageLogger.files.mtdCommonDataDDD = cms.untracked.PSet(
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

process.load('Configuration.Geometry.GeometryExtended2026D95_cff')

process.testBTL = cms.EDAnalyzer("TestMTDIdealGeometry",
                               ddTopNodeName = cms.untracked.string('BarrelTimingLayer')
                               )

process.testETL = cms.EDAnalyzer("TestMTDIdealGeometry",
                               ddTopNodeName = cms.untracked.string('EndcapTimingLayer')
                               )

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(cms.wait(process.testBTL)+process.testETL)
