# The following comments couldn't be translated into the new config version:

#      { string record = "HcalElectronicsMapRcd" string tag = "official_emap_16x_9Nov07" },

import FWCore.ParameterSet.Config as cms

from EventFilter.HcalRawToDigi.HcalRawToDigi_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi import *
from DQM.L1TMonitor.L1THCALTPG_cfi import *
from DQM.L1TMonitor.L1THCALTPGXAna_cfi import *
hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

hcalConditions = cms.ESSource("PoolDBESSource",
    #    string connect = "frontier://(serverurl=http://frontier1.cms:8000/FrontierOn)(serverurl=http://frontier2.cms:8000/FrontierOn)(retrieve-ziplevel=0)/CMS_COND_ON_170_HCAL"
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0)
    ),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('HcalPedestalsRcd'),
        tag = cms.string('hcal_pedestals_fC_gren')
    ), cms.PSet(
        record = cms.string('HcalElectronicsMapRcd'),
        tag = cms.string('emap_temp_slb_fix_2Dec07')
    ), cms.PSet(
        record = cms.string('HcalGainsRcd'),
        tag = cms.string('hcal_gains_v1')
    ), cms.PSet(
        record = cms.string('HcalQIEDataRcd'),
        tag = cms.string('qie_normalmode_v3')
    ), cms.PSet(
        record = cms.string('HcalPedestalWidthsRcd'),
        tag = cms.string('hcal_widths_fC_gren')
    )),
    connect = cms.string('frontier://Frontier/CMS_COND_ON_170_HCAL'),
    siteLocalConfig = cms.untracked.bool(False)
)

es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    toGet = cms.untracked.vstring('GainWidths', 'channelQuality')
)

p = cms.Path(hcalDigis*horeco*hfreco*hbhereco*l1thcaltpg*l1thcaltpgxana)

