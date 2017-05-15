import FWCore.ParameterSet.Config as cms
from CondCore.CondDB.CondDB_cfi import *
import CondTools.Ecal.conddb_init as conddb_init

process = cms.Process("ESAlignmentPopulator")

#LogDebug output is enabled: remember the USER_CXXFLAGS="-DEDM_ML_DEBUG" compilation flag to see it.
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring("cout"),
                                    debugModules = cms.untracked.vstring("*"),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string("DEBUG"))
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
                                          toPut = cms.VPSet(cms.PSet(record = cms.string("ESAlignmentRcd"),
                                                                     tag = cms.string(conddb_init.options.destinationTag)
                                                                     )
                                                            )
                                          )

process.popConESAlignment = cms.EDAnalyzer("EcalAlignmentPopConBTransitionAnalyzer",
                                                    SinceAppendMode = cms.bool(True),
                                                    record = cms.string("ESAlignmentRcd"),
                                                    Source = cms.PSet(BTransition = cms.PSet(CondDBConnection, #We write and read from the same DB
                                                                                             runNumber = cms.uint64(conddb_init.options.runNumber),
                                                                                             tagForRunInfo = cms.string(conddb_init.options.tagForRunInfo),
                                                                                             tagForBOff = cms.string(conddb_init.options.tagForBOff),
                                                                                             tagForBOn = cms.string(conddb_init.options.tagForBOn),
                                                                                             currentThreshold = cms.untracked.double(conddb_init.options.currentThreshold)
                                                                                             )
                                                                      ),
                                                    loggingOn = cms.untracked.bool(True),
                                                    targetDBConnectionString = cms.untracked.string("")
                                                    )

process.p = cms.Path(process.popConESAlignment)
