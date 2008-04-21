import FWCore.ParameterSet.Config as cms

GlobalTag = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    connect = cms.string('frontier://FrontierInt/CMS_COND_20X_GLOBALTAG'), ##FrontierInt/CMS_COND_20X_GLOBALTAG"

    globaltag = cms.untracked.string('IDEAL::All'),
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
)


