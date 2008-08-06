import FWCore.ParameterSet.Config as cms

process = cms.Process("DQMOfflineRealData")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('SiStripZeroSuppression', 
        'SiStripMonitorDigi', 
        'SiStripMonitorCluster', 
        'SiStripMonitorTrackSim', 
        'MonitorTrackResidualsSim'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    destinations = cms.untracked.vstring('cout')
)

#-------------------------------------------------
# Magnetic Field
#-------------------------------------------------
process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")

#-------------------------------------------------
# Geometry
#-------------------------------------------------
process.load("Configuration.StandardSequences.Geometry_cff")

#-------------------------------------------------
# Calibration
#-------------------------------------------------
import CalibTracker.Configuration.Common.PoolDBESSource_cfi
process.siStripCond = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
process.siStripCond.toGet = cms.VPSet(
    cms.PSet(record = cms.string('SiStripPedestalsRcd'),  tag = cms.string('SiStripPedestals_TKCC_21X_v3_hlt')), 
    cms.PSet(record = cms.string('SiStripNoisesRcd'),     tag = cms.string('SiStripNoise_TKCC_21X_v3_hlt')), 
    cms.PSet(record = cms.string('SiStripBadChannelRcd'), tag = cms.string('SiStripBadChannel_TKCC_21X_v3_hlt')), 
    cms.PSet(record = cms.string('SiStripFedCablingRcd'), tag = cms.string('SiStripFedCabling_TKCC_21X_v3_hlt'))
  )
process.siStripCond.connect = 'oracle://cms_orcoff_prod/CMS_COND_21X_STRIP'
process.siStripCond.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'

process.load("CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi")
process.SiStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
    cms.PSet(record = cms.string('SiStripDetCablingRcd'), tag = cms.string('')), 
    cms.PSet(record = cms.string('SiStripBadChannelRcd'), tag = cms.string(''))
)

process.sistripconn = cms.ESProducer("SiStripConnectivity")

process.load("CalibTracker.Configuration.SiStripGain.SiStripGain_Fake_cff")

process.load("CalibTracker.Configuration.SiStripLorentzAngle.SiStripLorentzAngle_Fake_cff")

process.load("CalibTracker.Configuration.SiPixelLorentzAngle.SiPixelLorentzAngle_Fake_cff")

process.load("CalibTracker.Configuration.TrackerAlignment.TrackerAlignment_Fake_cff")

#If Frontier is used in xdaq environment use the following service
#    service = SiteLocalConfigService {}

#-----------------------
# Reconstruction Modules
#-----------------------
process.load("DQM.SiStripMonitorClient.RecoForDQM_cff")

#--------------------------
# DQM
#--------------------------
process.load("DQM.SiStripMonitorClient.SiStripDQMOffline_cff")

process.p = cms.Path(process.RecoForDQM*process.SiStripDQMOffRealData)

process.AdaptorConfig = cms.Service("AdaptorConfig")

#-------------------------
# Input Events
#-------------------------
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/data/CRUZET3/Cosmics/RAW/v1/000/051/490/02D59D05-4151-DD11-9E79-001617DBD5AC.root', 
        '/store/data/CRUZET3/Cosmics/RAW/v1/000/051/490/02E220B3-4451-DD11-8471-000423D98868.root', 
        '/store/data/CRUZET3/Cosmics/RAW/v1/000/051/490/041473E8-4C51-DD11-88BC-0019DB29C5FC.root', 
        '/store/data/CRUZET3/Cosmics/RAW/v1/000/051/490/049C5899-3851-DD11-B6DD-001617DBD230.root', 
        '/store/data/CRUZET3/Cosmics/RAW/v1/000/051/490/04C4B6BF-4A51-DD11-8E21-001617E30F50.root', 
        '/store/data/CRUZET3/Cosmics/RAW/v1/000/051/490/06E775CA-4751-DD11-AC1C-000423D94990.root', 
        '/store/data/CRUZET3/Cosmics/RAW/v1/000/051/490/0EC99DCB-4851-DD11-9A47-000423D9A212.root', 
        '/store/data/CRUZET3/Cosmics/RAW/v1/000/051/490/1000EF79-4C51-DD11-A902-0030487C6090.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)


