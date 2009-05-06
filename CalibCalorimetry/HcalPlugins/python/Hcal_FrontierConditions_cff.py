import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *
hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

es_pool = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    timetype = cms.string('runnumber'),
    connect = cms.string('frontier://FrontierDev/CMS_COND_HCAL'), ##FrontierDev/CMS_COND_HCAL"
    authenticationMethod = cms.untracked.uint32(0),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('HcalPedestalsRcd'),
            tag = cms.string('hcal_pedestals_fC_v6_mc')
        ), 
        cms.PSet(
            record = cms.string('HcalPedestalWidthsRcd'),
            tag = cms.string('hcal_widths_fC_v6_mc')
        ), 
        cms.PSet(
            record = cms.string('HcalGainsRcd'),
            tag = cms.string('hcal_gains_v3.01_physics_mc')
        ), 
        cms.PSet(
            record = cms.string('HcalQIEDataRcd'),
            tag = cms.string('qie_normalmode_v6.01')
        ), 
        cms.PSet(
            record = cms.string('HcalChannelQualityRcd'),
            tag = cms.string('hcal_channelStatus_trivial_mc')
        ), 
        cms.PSet(
            record = cms.string('HcalRespCorrsRcd'),
            tag = cms.string('hcal_respcorr_trivial_v1.01_mc')
        ), 
        cms.PSet(
            record = cms.string('HcalTimeCorrsRcd'),
            tag = cms.string('hcal_timecorr_trivial_v1.00_mc')
        ), 
        cms.PSet(
            record = cms.string('HcalL1TriggerObjectsRcd'),
            tag = cms.string('hcal_L1TriggerObjects_trivial_mc')
        ), 
        cms.PSet(
            record = cms.string('HcalElectronicsMapRcd'),
            tag = cms.string('official_emap_v7.00')
        )
     )
)

es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    toGet = cms.untracked.vstring('GainWidths', 
        'ZSThresholds')
)

