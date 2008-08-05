# The following comments couldn't be translated into the new config version:

#Frontier/CMS_COND_20X_STRIP"
import FWCore.ParameterSet.Config as cms

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
siStripCond = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
sistripconn = cms.ESProducer("SiStripConnectivity")

siStripCond.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiStripFedCablingRcd'),
    tag = cms.string('SiStripFedCabling_TKCC_20X_v3_hlt')
), 
    cms.PSet(
        record = cms.string('SiStripNoisesRcd'),
        tag = cms.string('SiStripNoise_TKCC_20X_v3_hlt')
    ), 
    cms.PSet(
        record = cms.string('SiStripPedestalsRcd'),
        tag = cms.string('SiStripPedestals_TKCC_20X_v3_hlt')
    ))
siStripCond.connect = 'frontier://Frontier/CMS_COND_20X_STRIP'
siStripCond.DBParameters.authenticationPath = ''

