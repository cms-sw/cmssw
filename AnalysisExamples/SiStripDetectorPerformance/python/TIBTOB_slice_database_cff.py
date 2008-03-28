import FWCore.ParameterSet.Config as cms

PoolDBESSource = cms.ESSource("PoolDBESSource",
    loadBlobStreamer = cms.untracked.bool(True),
    loadAll = cms.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripPedestalsRcd'),
        tag = cms.string('SiStripPedNoise_TIBTOB_v1_p')
    ), cms.PSet(
        record = cms.string('SiStripNoisesRcd'),
        tag = cms.string('SiStripPedNoise_TIBTOB_v1_n')
    ), cms.PSet(
        record = cms.string('SiStripFedCablingRcd'),
        tag = cms.string('SiStripCabling_TIBTOB_v1')
    )),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        loadBlobStreamer = cms.untracked.bool(True)
    ),
    catalog = cms.untracked.string('relationalcatalog_oracle://cms_orcoff_int2r/CMS_COND_GENERAL'),
    timetype = cms.string('runnumber'),
    connect = cms.string('oracle://cms_orcoff_int2r/CMS_COND_STRIP'),
    authenticationMethod = cms.untracked.uint32(1)
)


