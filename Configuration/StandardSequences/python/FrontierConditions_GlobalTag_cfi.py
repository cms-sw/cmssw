import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *

GlobalTag = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    connect = cms.string('frontier://FrontierInt/CMS_COND_21X_GLOBALTAG'), ##FrontierProd/CMS_COND_21X_GLOBALTAG"

    globaltag = cms.untracked.string('IDEAL_V1::All'),
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
)
