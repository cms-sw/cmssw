import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMStore_cfg import *
from CondCore.CondDB.CondDB_cfi import *
CondDBReference = CondDB.clone(connect = cms.string('oracle://cms_orcoff_prep/CMS_COND_TEMP'))
CondDBReference.DBParameters.messageLevel = cms.untracked.int32(1) #3 for high verbosity

ReferenceRetrieval = cms.ESSource("PoolDBESSource",
                                  CondDBReference,
                                  toGet = cms.VPSet(cms.PSet(record = cms.string('DQMReferenceHistogramRootFileRcd'),
                                                             tag = cms.string('DQM_Cosmics_prompt')
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
