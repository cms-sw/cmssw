import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("Geometry.TrackerCommonData.testGeometryXML_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring('*'),
    files = cms.untracked.PSet(
        debug = cms.untracked.PSet(
            DEBUG = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            INFO = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            TECGeom = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            TrackerGeom = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            threshold = cms.untracked.string('DEBUG')
        )
    )
)

process.m = cms.EDAnalyzer("PerfectGeometryAnalyzer",
    dumpGeoHistory = cms.untracked.bool(False),
    dumpPosInfo    = cms.untracked.bool(False),
    dumpSpecs      = cms.untracked.bool(False)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.p1 = cms.Path(process.m)
