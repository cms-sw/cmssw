import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("Geometry.TrackerCommonData.testGeometryXML_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('debug'),
    categories   = cms.untracked.vstring('TrackerGeom', 'TECGeom'),
    debugModules = cms.untracked.vstring('*'),
    debug        = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        TrackerGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        TECGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
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
