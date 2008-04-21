# The following comments couldn't be translated into the new config version:

# not used in reconstruction

import FWCore.ParameterSet.Config as cms

hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

es_pool = cms.ESSource("PoolDBESSource",
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('HcalPedestalsRcd'),
        tag = cms.string('pedestals_mtcc2_v1')
    ), 
        cms.PSet(
            record = cms.string('HcalGainsRcd'),
            tag = cms.string('hcal_gains_hardcoded_v1')
        ), 
        cms.PSet(
            record = cms.string('HcalQIEDataRcd'),
            tag = cms.string('hcal_qie_hardcoded_v1')
        ), 
        cms.PSet(
            record = cms.string('HcalElectronicsMapRcd'),
            tag = cms.string('emap_mtcc2_v4')
        ), 
        cms.PSet(
            record = cms.string('HcalPedestalWidthsRcd'),
            tag = cms.string('hcal_pwidths_hardcoded_v1')
        ), 
        cms.PSet(
            record = cms.string('HcalGainWidthsRcd'),
            tag = cms.string('hcal_gwidths_hardcoded_v1')
        )),
    connect = cms.string('frontier://cms_conditions_data/CMS_COND_HCAL') ##cms_conditions_data/CMS_COND_HCAL"

)

es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    toGet = cms.untracked.vstring('channelQuality')
)


