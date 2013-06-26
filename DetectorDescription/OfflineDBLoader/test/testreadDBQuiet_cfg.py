import FWCore.ParameterSet.Config as cms

process = cms.Process("DBGeometryTest")
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )
process.source = cms.Source("EmptySource")

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                         process.CondDBSetup,
                                         loadAll = cms.bool(True),
                                         toGet = cms.VPSet(cms.PSet(
                                            record = cms.string('IdealGeometryRecord'),
                                            tag = cms.string('IdealGeometry01')
                                         )),
                                         DBParameters = cms.PSet(
                                            messageLevel = cms.untracked.int32(9),
                                            authenticationPath = cms.untracked.string('.')
                                         ),
                                         catalog = cms.untracked.string('file:PoolFileCatalog.xml'),
                                         timetype = cms.string('runnumber'),
                                         connect = cms.string('sqlite_file:testIdeal.db')
                                      )

process.myprint = cms.OutputModule("AsciiOutputModule")

process.Timing = cms.Service("Timing")

process.prod = cms.EDAnalyzer("PerfectGeometryAnalyzer",
                                  dumpPosInfo = cms.untracked.bool(False),
                                  dumpSpecs = cms.untracked.bool(False),
                                  dumpGeoHistory = cms.untracked.bool(False)
                              )

process.p1 = cms.Path(process.prod)
process.e1 = cms.EndPath(process.myprint)

