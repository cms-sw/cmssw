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
      '/store/data/CRAFT09/Cosmics/RAW/v1/000/110/998/D2947736-8189-DE11-A17E-000423D94C68.root',
#      '/store/data/CRAFT09/Cosmics/RAW/v1/000/110/998/DE49D0FC-2C8A-DE11-BA21-000423D6A6F4.root'
    )
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

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
process.load("Configuration.StandardSequences.Geometry_cff")

#--------------------------
# Calibration
#--------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR09_31X_V6P::All"

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
process.p = cms.Path(process.RawToDigi_woGCT*process.reconstructionCosmics*process.SiStripDQMTest)


process.outpath = cms.EndPath(process.myOut)
