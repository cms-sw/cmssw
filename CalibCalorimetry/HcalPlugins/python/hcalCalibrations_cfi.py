import FWCore.ParameterSet.Config as cms

hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

es_pool = cms.ESSource("PoolDBESSource",
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('HcalPedestalsRcd'),
        tag = cms.string('hcal_pedestals_fC_v1')
    ), cms.PSet(
        record = cms.string('HcalGainsRcd'),
        tag = cms.string('hcal_gains_v1')
    ), cms.PSet(
        record = cms.string('HcalQIEDataRcd'),
        tag = cms.string('qie_normalmode_v3')
    ), cms.PSet(
        record = cms.string('HcalElectronicsMapRcd'),
        tag = cms.string('hfplusmap_v2')
    ), cms.PSet(
        record = cms.string('HcalPedestalWidthsRcd'),
        tag = cms.string('hcal_widths_fC_v1')
    )),
    connect = cms.string('frontier://cms_conditions_data/CMS_COND_HCAL')
)

es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    toGet = cms.untracked.vstring('GainWidths', 'channelQuality')
)


