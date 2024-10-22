import FWCore.ParameterSet.Config as cms
from CondCore.CondDB.CondDB_cfi import *
import CondTools.Ecal.conddb_init as conddb_init

process = cms.Process("ESRecHitRatioCutsPopulator")

process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring("cout"),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string("INFO"))
                                    )

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(conddb_init.options.runNumber),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(conddb_init.options.runNumber),
                            interval = cms.uint64(1)
)

CondDBConnection = CondDB.clone(connect = cms.string(conddb_init.options.destinationDatabase))

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          CondDBConnection,
                                          toPut = cms.VPSet(cms.PSet(record = cms.string("ESRecHitRatioCutsRcd"),
                                                                     tag = cms.string(conddb_init.options.destinationTag)
                                                                     )
                                                            )
                                          )

process.popConESRecHitRatioCuts = cms.EDAnalyzer("ESRecHitRatioCutsPopConTransitionAnalyzer",
                                                    SinceAppendMode = cms.bool(True),
                                                    record = cms.string("ESRecHitRatioCutsRcd"),
                                                    Source = cms.PSet(ESTransition = cms.PSet(CondDBConnection, #We write and read from the same DB
                                                                                             runNumber = cms.uint64(conddb_init.options.runNumber),
                                                                                             tagForRunInfo = cms.string(conddb_init.options.tagForRunInfo),
                                                                                             ESGain = cms.string(conddb_init.options.ESGain),
                                                                                             ESLowGainTag = cms.string(conddb_init.options.ESLowGainTag),
                                                                                             ESHighGainTag = cms.string(conddb_init.options.ESHighGainTag)                                                                                                   )
                                                                      ),
                                                    loggingOn = cms.untracked.bool(True),
                                                    targetDBConnectionString = cms.untracked.string("")
                                                    )

process.p = cms.Path(process.popConESRecHitRatioCuts)
