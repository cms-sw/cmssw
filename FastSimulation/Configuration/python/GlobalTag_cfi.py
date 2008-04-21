import FWCore.ParameterSet.Config as cms

GlobalTag = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    connect = cms.string('frontier://FrontierProd/CMS_COND_20X_GLOBALTAG'), ##FrontierProd/CMS_COND_20X_GLOBALTAG"

    globaltag = cms.untracked.string('IDEAL::All'),
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
)


