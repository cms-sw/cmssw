import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *

GlobalTag = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    connect = cms.string('frontier://FrontierDev/CMS_COND_GLOBALTAG'), ##FrontierProd/CMS_COND_21X_GLOBALTAG"

    globaltag = cms.untracked.string('IDEAL::All'),
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
)
