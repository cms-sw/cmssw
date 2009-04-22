import FWCore.ParameterSet.Config as cms
from CondCore.DBCommon.CondDBSetup_cfi import *
LumiESSource = cms.ESSource( "PoolDBESSource",
  CondDBSetup,
  connect = cms.string('sqlite_fip:CondCore/SQLiteData/data/lumi.db'),
  BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
)
from RecoLuminosity.LumiProducer.lumiProducer_cfi import *
