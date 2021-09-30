import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMStore_cfg import *

dqmRefHistoRootFileGetter=cms.EDAnalyzer("DQMReferenceHistogramRootFileEventSetupAnalyzer") 

## from CondCore.CondDB.CondDB_cfi import *
## CondDBReference = CondDB.clone(connect = cms.string('sqlite_file:DQMReferenceHistogramTest.db'))
## CondDBReference.DBParameters.messageLevel = cms.untracked.int32(1) #3 for high verbosity
## ReferenceRetrieval = cms.ESSource("PoolDBESSource",
##                                   CondDBReference,
##                                   toGet = cms.VPSet(cms.PSet(record = cms.string('DQMReferenceHistogramRootFileRcd'),
##                                                              tag = cms.string('ROOTFILE_DQM_Test10')
##                                                              )
##                                                     )
##                                   )
##
## RecordDataGetter = cms.EDAnalyzer("EventSetupRecordDataGetter",
##                              toGet = cms.VPSet(cms.PSet(record = cms.string('DQMReferenceHistogramRootFileRcd'),
##                                                         data = cms.vstring('ROOTFILE_DQM_Test10')
##                                                         )
##                                                ),
##                              verbose = cms.untracked.bool(False)
##                              )
