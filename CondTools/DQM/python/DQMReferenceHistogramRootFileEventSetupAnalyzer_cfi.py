import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMStore_cfg import *
from CondCore.DBCommon.CondDBSetup_cfi import *

ReferenceRetrieval = cms.ESSource("PoolDBESSource",
                                  CondDBSetup,
                                  connect = cms.string('sqlite_file:DQMReferenceHistogramTest.db'),
                                  BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                  messageLevel = cms.untracked.int32(1), #3 for high verbosity
                                  timetype = cms.string('runnumber'),
                                  toGet = cms.VPSet(cms.PSet(record = cms.string('DQMReferenceHistogramRootFileRcd'),
                                                             tag = cms.string('ROOTFILE_DQM_Test10')
                                                             )
                                                    )
                                  )

## RecordDataGetter = cms.EDAnalyzer("EventSetupRecordDataGetter",
##                              toGet = cms.VPSet(cms.PSet(record = cms.string('DQMReferenceHistogramRootFileRcd'),
##                                                         data = cms.vstring('ROOTFILE_DQM_Test10')
##                                                         )
##                                                ),
##                              verbose = cms.untracked.bool(False)
##                              )
