import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.load("DetectorDescription.OfflineDBLoader.test.cmsIdealGeometryForWrite_cfi")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )
process.source = cms.Source("EmptyIOVSource",
                                lastValue = cms.uint64(1),
                                timetype = cms.string('runnumber'),
                                firstValue = cms.uint64(1),
                                interval = cms.uint64(1)
                            )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          DBParameters = cms.PSet(
                                             messageLevel = cms.untracked.int32(0),
                                             authenticationPath = cms.untracked.string('.')
                                             ),
                                          timetype = cms.untracked.string('runnumber'),
                                          connect = cms.string('sqlite_file:testIdeal.db'),
#                                              process.CondDBCommon,
                                          toPut = cms.VPSet(cms.PSet(
                                                                    record = cms.string('IdealGeometryRecord'),
                                                                    tag = cms.string('IdealGeometry01')
                                                            ))
                                          )

process.load = cms.EDAnalyzer("WriteOneGeometryFromXML",
                                rotNumSeed = cms.int32(0),
                                dumpSpecs = cms.untracked.bool(True),
                                dumpGeoHistory = cms.untracked.bool(True),
                                dumpPosInfo = cms.untracked.bool(True)
                            )

process.myprint = cms.OutputModule("AsciiOutputModule")

process.p1 = cms.Path(process.load)
process.ep = cms.EndPath(process.myprint)
process.CondDBCommon.connect = 'sqlite_file:testIdeal.db'
process.CondDBCommon.DBParameters.messageLevel = 0
process.CondDBCommon.DBParameters.authenticationPath = './'
