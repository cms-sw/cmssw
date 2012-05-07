import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *

GlobalTag = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    connect = cms.string('frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'),
    globaltag = cms.string('UNSPECIFIED::All'),
    toGet = cms.VPSet( ),   # hook to override or add single payloads
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
)
