# The following comments couldn't be translated into the new config version:

#                    { string record = "SiStripApvGainRcd" string tag = "SiStripGain_StartUp_20X_mc" },
#                    { string record = "SiStripLorentzAngleRcd" string tag = "SiStripLorentzAngle_StartUp_20X_mc" },

#Frontier/CMS_COND_21X_STRIP"
import FWCore.ParameterSet.Config as cms

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
siStripCond = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
from DQM.SiStripMonitorSummary.tagsQuality_cfi import *
sistripconn = cms.ESProducer("SiStripConnectivity")

siStripCond.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiStripFedCablingRcd'),
    tag = cms.string('SiStripFedCabling_TKCC_21X_v3_hlt')
), 
    cms.PSet(
        record = cms.string('SiStripNoisesRcd'),
        tag = cms.string('SiStripNoise_TKCC_21X_v3_hlt')
    ), 
    cms.PSet(
        record = cms.string('SiStripPedestalsRcd'),
        tag = cms.string('SiStripPedestals_TKCC_21X_v3_hlt')
    ), 
    cms.PSet(
        record = cms.string('SiStripThresholdRcd'),
        tag = cms.string('SiStripThreshold_TKCC_21X_v3_hlt')
    ))
siStripCond.connect = 'frontier://Frontier/CMS_COND_21X_STRIP'
siStripCond.DBParameters.authenticationPath = ''


