import FWCore.ParameterSet.Config as cms

process = cms.Process("SiStrpDQMTier0Test")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siStripDigis', 
                                         'siStripClusters', 
                                         'siStripZeroSuppression', 
                                         'SiStripClusterizer'),
    cout = cms.untracked.PSet(threshold = cms.untracked.string('ERROR')),
    destinations = cms.untracked.vstring('cout')
)

#----------------------------
# Event Source
#-----------------------------
process.source = cms.Source("PoolSource",
     fileNames = cms.untracked.vstring(
#     '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/365/0A92FA00-01AB-DD11-A6AF-001617DBD288.root'
     '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/365/8C367B78-05AB-DD11-8ADF-001617C3B79A.root'
     )
)                            
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(2000))

#----------------------------
# DQM Environment
#-----------------------------
process.load("DQMServices.Core.DQM_cfg")
#process.DQMStore.referenceFileName = '/home/dqmdevlocal/reference/sistrip_reference.root'

process.load("DQMServices.Components.DQMEnvironment_cfi")

#----------------------------
# DQM Playback Environment
#-----------------------------
process.load("DQM.Integration.test.environment_playback_cfi")
process.dqmEnv.subSystemFolder    = "SiStrip"

#-----------------------------
# Magnetic Field
#-----------------------------
process.load("Configuration.StandardSequences.MagneticField_0T_cff")
process.prefer("VolumeBasedMagneticFieldESProducer")

#-------------------------------------------------
# GEOMETRY
#-------------------------------------------------
process.load("Configuration.StandardSequences.Geometry_cff")

#--------------------------
# Calibration
#--------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect = "frontier://FrontierInt/CMS_COND_30X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT_30X::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

#-----------------------
#  Reconstruction Modules
#-----------------------
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")

#--------------------------
# Strip DQM Sources and DQ
#--------------------------
process.load("DQM.SiStripMonitorClient.SiStripDQMTier0GlobalRun_cff")

#----------------------- 
# output module
#----------------------
process.myOut = cms.OutputModule("PoolOutputModule",
                                  fileName = cms.untracked.string('sistrip_reco2.root'),
                                  outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*')
                                )
#--------------------------
# Scheduling
#--------------------------
process.p = cms.Path(process.RawToDigi_woGCT*process.reconstructionCosmics*process.SiStripDQMTest)

process.outpath = cms.EndPath(process.myOut)
