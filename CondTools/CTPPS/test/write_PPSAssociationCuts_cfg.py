##### configuration #####
output_conditions = 'sqlite_file:association_config.db'  # output database
run_number = 1  # beginning of the IOV
db_tag = 'PPSAssociationCuts_test'  # database tag
product_instance_label = 'db_test_label'  # ES product label
#########################

import FWCore.ParameterSet.Config as cms

process = cms.Process("writePPSAssociationCuts")

# Message Logger

process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(
                                        threshold = cms.untracked.string('INFO')
                                    )
                                    )

# Load CondDB service
process.load("CondCore.CondDB.CondDB_cfi")

# output database
process.CondDB.connect = output_conditions

# A data source must always be defined. We don't need it, so here's a dummy one.
process.source = cms.Source("EmptyIOVSource",
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(run_number),
                            lastValue = cms.uint64(run_number),
                            interval = cms.uint64(1)
                            )

# output service
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(
                                              record = cms.string('PPSAssociationCutsRcd'),
                                              tag = cms.string(db_tag)
                                          ))
                                          )

# ESSource

import CalibPPS.ESProducers.ppsAssociationCuts_non_DB_cff as ac
ac.use_single_infinite_iov_entry(ac.ppsAssociationCutsESSource,ac.p2018)

process.ppsAssociationCutsESSource = ac.ppsAssociationCutsESSource
process.ppsAssociationCutsESSource.appendToDataLabel = cms.string('product_instance_label')




# DB object maker
process.config_writer = cms.EDAnalyzer("WritePPSAssociationCuts",
                                       record = cms.string('PPSAssociationCutsRcd'),
                                       loggingOn = cms.untracked.bool(True),
                                       SinceAppendMode = cms.bool(True),
                                       Source = cms.PSet(
                                           IOVRun = cms.untracked.uint32(1)
                                       ),
                                       label = cms.string("product_instance_label")
                                       )

process.path = cms.Path(process.config_writer)
