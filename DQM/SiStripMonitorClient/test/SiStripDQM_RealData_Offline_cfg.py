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
process.load("Configuration.StandardSequences.MagneticField_0T_cff")
process.prefer("VolumeBasedMagneticFieldESProducer")

#-------------------------------------------------
# Geometry
#-------------------------------------------------
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-------------------------------------------------
# Calibration
#-------------------------------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect = "frontier://FrontierInt/CMS_COND_30X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT_30X::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

#-----------------------
# Reconstruction Modules
#-----------------------
process.load("DQM.SiStripMonitorClient.RecoForDQM_Cosmic_cff")

#--------------------------
# DQM
#--------------------------
process.load("DQM.SiStripMonitorClient.SiStripDQMOffline_cff")

process.p = cms.Path(process.RecoForDQMCosmic*process.SiStripDQMOffRealData)

process.AdaptorConfig = cms.Service("AdaptorConfig")

#-------------------------
# Input Events
#-------------------------
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       '/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/498/22957003-566D-DD11-994B-001617C3B654.root',
       '/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/498/2E0A2187-556D-DD11-8FD5-001617C3B70E.root',
       '/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/498/3AA5D2BD-576D-DD11-A629-000423D6CA72.root',
       '/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/498/A0A8861B-5A6D-DD11-A686-000423D986A8.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)


