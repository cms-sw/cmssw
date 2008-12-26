import FWCore.ParameterSet.Config as cms

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
siStripCond = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()

####from DQM.SiStripMonitorSummary.Template_tagsQuality_cfi import *
sistripconn = cms.ESProducer("SiStripConnectivity")

siStripCond.toGet = cms.VPSet(
    cms.PSet(
    record = cms.string('SiStripFedCablingRcd'),
    tag = cms.string('insert_FedCablingTag')
    ), 
    cms.PSet( 
        record = cms.string('SiStripNoisesRcd'),
        tag = cms.string('insert_NoiseTag')
    ), 
    cms.PSet(
        record = cms.string('SiStripPedestalsRcd'),
        tag = cms.string('insert_PedestalTag')
    ),
    cms.PSet(
        record = cms.string('SiStripApvGainRcd'),
        tag = cms.string('SiStripGain_Ideal_21X')
    ),
    cms.PSet(
        record = cms.string('SiStripLorentzAngleRcd'),
        tag = cms.string('SiStripLorentzAngle_Ideal_21X')
    ),     
    cms.PSet(
        record = cms.string('SiStripThresholdRcd'),
        tag = cms.string('insert_ThresholdTag')
    ))


    
siStripCond.connect = 'frontier://cmsfrontier.cern.ch:8000/FrontierProd/insertAccount'
###siStripCond.connect = 'oracle://cms_orcoff_prod/CMS_COND_21X_STRIP'
##siStripCond.connect = 'frontier://Frontier/CMS_COND_21X_STRIP'


###siStripCond.DBParameters.authenticationPath = '' 
###siStripCond.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb' 


