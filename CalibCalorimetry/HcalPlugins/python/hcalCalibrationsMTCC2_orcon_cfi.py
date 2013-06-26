# The following comments couldn't be translated into the new config version:

# not used in reconstruction

import FWCore.ParameterSet.Config as cms

# for this to work need the following:
# setenv POOL_CATALOG relationalcatalog_oracle://orcon/CMS_COND_GENERAL
# setenv CORAL_AUTH_PATH /afs/cern.ch/cms/DB/conddb
hcal_db_producer = cms.ESProducer("HcalDbProducer")

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
            tag = cms.string('emap_mtcc2_v3')
        ), 
        cms.PSet(
            record = cms.string('HcalPedestalWidthsRcd'),
            tag = cms.string('hcal_pwidths_hardcoded_v1')
        ), 
        cms.PSet(
            record = cms.string('HcalGainWidthsRcd'),
            tag = cms.string('hcal_gwidths_hardcoded_v1')
        )),
    connect = cms.string('oracle://orcon/CMS_COND_HCAL'), ##orcon/CMS_COND_HCAL"

    authenticationMethod = cms.untracked.uint32(1)
)

es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    toGet = cms.untracked.vstring('channelQuality')
)


