import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'sqlite_file:ints.db'

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('Run'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          timetype = cms.untracked.string('Run'),
                                          outOfOrder = cms.untracked.bool(True),
                                          toPut = cms.VPSet( cms.PSet(record = cms.string('oneInt'),
                                                                      tag = cms.string('OneInt'),
                                                                      timetype = cms.untracked.string('Run'),
                                                                      outOfOrder = cms.untracked.bool(False)
                                                                      )
                                                             )
                                          )

process.mytest = cms.EDAnalyzer("writeInt",
                                Number=cms.int32(_CurrentRun_)
                                )

process.p = cms.Path(process.mytest)

