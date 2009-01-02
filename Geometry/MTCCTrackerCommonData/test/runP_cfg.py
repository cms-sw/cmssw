import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("Geometry.TrackerSimData.trackerSimGeometry_MTCCXML_cfi")

process.p = cms.EDAnalyzer("AsciiOutputModule")
                           
process.m = cms.EDAnalyzer("PerfectGeometryAnalyzer",
    dumpGeoHistory = cms.untracked.bool(True),
    dumpPosInfo    = cms.untracked.bool(True),
    dumpSpecs      = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.p1 = cms.Path(process.p*process.m)
