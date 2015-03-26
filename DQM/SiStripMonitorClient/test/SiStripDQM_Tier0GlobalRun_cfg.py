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
## CRAFT08   
#       '/store/data/Commissioning08/Cosmics/RAW/v1/000/067/838/006945C8-40A5-DD11-BD7E-001617DBD556.root'
#      '/store/data/Commissioning08/Cosmics/RAW/v1/000/067/838/00BAAF73-52A5-DD11-9351-001D09F23A84.root'

## CRAFT09
       '/store/data/CRAFT09/Cosmics/RAW/v1/000/110/998/001404E1-0F8A-DE11-ADB3-000423D99EEE.root',
#      '/store/data/CRAFT09/Cosmics/RAW/v1/000/110/998/002174A8-E989-DE11-8B4D-000423D6CA42.root'
      
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

process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')

#-------------------------------------------------
# GEOMETRY
#-------------------------------------------------
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#--------------------------
# Calibration
#--------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# CRAFT08
#process.GlobalTag.globaltag = CRAFT0831X_V1::All
# CRAFT09
process.GlobalTag.globaltag = "CRAFT09_R_V3::All"

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
                                  fileName = cms.untracked.string('sistrip_reco1.root'),
                                  outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*')
                                )
#--------------------------
# Scheduling
#--------------------------
process.p = cms.Path(process.siPixelDigis*process.siStripDigis*process.trackerCosmics*process.SiStripDQMTest)


process.outpath = cms.EndPath(process.myOut)
