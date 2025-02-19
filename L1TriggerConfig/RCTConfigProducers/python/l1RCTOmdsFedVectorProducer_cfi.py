import FWCore.ParameterSet.Config as cms

l1RCTOmdsFedVectorProducer = cms.ESProducer("L1RCTOmdsFedVectorProducer",
                                            connectionString = cms.string("oracle://cms_orcoff_prod/CMS_RUNINFO"),
                                            authpath = cms.string("/afs/cern.ch/cms/DB/conddb"),
                                            tableToRead = cms.string("RUNSESSION_PARAMETER")
                                            )

