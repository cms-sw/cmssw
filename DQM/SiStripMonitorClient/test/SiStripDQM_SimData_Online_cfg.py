import FWCore.ParameterSet.Config as cms

process = cms.Process("DQMOnlineSimData")
#-------------------------------------------------
# Message Logger
#-------------------------------------------------
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('ERROR')
    ),
    debugModules = cms.untracked.vstring(
        'SiStripZeroSuppression', 
        'SiStripMonitorDigi', 
        'SiStripMonitorCluster', 
        'SiStripMonitorTrackSim', 
        'MonitorTrackResidualsSim'
    )
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
# CALIBRATION
#-------------------------------------------------
process.load("CalibTracker.Configuration.Tracker_FakeConditions_cff")

#If Frontier is used in xdaq environment use the following service
#    service = SiteLocalConfigService {}
#-----------------------
#  Reconstruction Modules
#-----------------------
process.load("DQM.SiStripMonitorClient.RecoForDQM_Sim_cff")

#--------------------------
# DQM
#--------------------------
process.load("DQM.SiStripMonitorClient.SiStripDQMOffline_cff")

#----------------------------
#### Scheduling
#-----------------------------
process.p = cms.Path(process.RecoModulesForSimData*process.SiStripDQMOffSimData)

process.outP = cms.OutputModule("AsciiOutputModule")
process.ep = cms.EndPath(process.outP)

process.AdaptorConfig = cms.Service("AdaptorConfig")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/7/21/RelVal-RelValQCD_Pt_80_120-1216579576-STARTUP_V4-2nd/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579576-STARTUP_V4-2nd-unmerged/0000/0A81C8A7-6E57-DD11-8B07-00161757BF42.root', 
        '/store/relval/2008/7/21/RelVal-RelValQCD_Pt_80_120-1216579576-STARTUP_V4-2nd/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579576-STARTUP_V4-2nd-unmerged/0000/0CDA198B-7057-DD11-B23E-001617C3B77C.root', 
        '/store/relval/2008/7/21/RelVal-RelValQCD_Pt_80_120-1216579576-STARTUP_V4-2nd/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579576-STARTUP_V4-2nd-unmerged/0000/182EA0C2-7057-DD11-9B09-000423D9880C.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)



