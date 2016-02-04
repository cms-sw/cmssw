#-------------- EcnaSystemPythoModuleInsert_1 / beginning
import FWCore.ParameterSet.Config as cms

process = cms.Process("ECNA")

#-------------- Message Logger
process.MessageLogger = cms.Service("MessageLogger",
                                    suppressInfo = cms.untracked.vstring('ecalEBunpacker'),
                                    destinations = cms.untracked.vstring('last_job_INFOS.txt')
                                    )
#-------------- EcnaSystemPythoModuleInsert_1 / end
