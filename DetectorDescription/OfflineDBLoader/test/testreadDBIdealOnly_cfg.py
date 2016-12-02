import FWCore.ParameterSet.Config as cms

process = cms.Process("DBGeometryTest")
process.load('CondCore.CondDB.CondDB_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.source = cms.Source("EmptySource")

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                      connect = cms.string("sqlite_file:testIdeal.db"),
                                      toGet = cms.VPSet(cms.PSet(record = cms.string('IdealGeometryRecord'),
                                                                 tag = cms.string('IdealGeometry01'))
                                                        )
                                      )

process.myprint = cms.OutputModule("AsciiOutputModule")

process.Timing = cms.Service("Timing")

process.prod = cms.EDAnalyzer("PerfectGeometryAnalyzer",
                              dumpPosInfo = cms.untracked.bool(True),
                              dumpSpecs = cms.untracked.bool(True),
                              dumpGeoHistory = cms.untracked.bool(True)
                              )

process.p1 = cms.Path(process.prod)
process.e1 = cms.EndPath(process.myprint)
