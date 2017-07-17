import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMStore_cfg import *
from CondCore.CondDB.CondDB_cfi import *
CondDBReference = CondDB.clone(connect = cms.string('sqlite_file:ROOTFILE_Test.db'))
CondDBReference.DBParameters.messageLevel = cms.untracked.int32(1) #3 for high verbosity

ReferenceRetrieval = cms.ESSource("PoolDBESSource",
                                  CondDBReference,
                                  toGet = cms.VPSet(cms.PSet(record = cms.string('DQMReferenceHistogramRootFileRcd'),
                                                             tag = cms.string('ROOTFILE_Test')
                                                             )
                                                    )
                                  )

## RecordDataGetter = cms.EDAnalyzer("EventSetupRecordDataGetter",
##                              toGet = cms.VPSet(cms.PSet(record = cms.string('DQMReferenceHistogramRootFileRcd'),
##                                                         data = cms.vstring('DQM_Cosmics_prompt')
##                                                         )
##                                                ),
##                              verbose = cms.untracked.bool(False)
##                              )
