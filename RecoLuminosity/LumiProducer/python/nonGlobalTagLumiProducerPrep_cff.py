import FWCore.ParameterSet.Config as cms
from CondCore.DBCommon.CondDBSetup_cfi import *
LumiESSource = cms.ESSource( "PoolDBESSource",
  CondDBSetup,
  connect = cms.string('oracle://cms_orcoff_prep/CMS_COND_RUN_INFO'),
  BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
)
from RecoLuminosity.LumiProducer.lumiProducer_cfi import *
