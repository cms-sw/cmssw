import FWCore.ParameterSet.Config as cms

process = cms.Process("DBGeometryTest")

process.load("DetectorDescription.OfflineDBLoader.test.dumpns")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )
process.source = cms.Source("EmptySource")

process.myprint = cms.OutputModule("AsciiOutputModule")

process.prod = cms.EDAnalyzer("PerfectGeometryAnalyzer",
                              dumpPosInfo = cms.untracked.bool(False),
                              dumpSpecs = cms.untracked.bool(False),
                              dumpGeoHistory = cms.untracked.bool(False)
                              )

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.prod)
process.e1 = cms.EndPath(process.myprint)
