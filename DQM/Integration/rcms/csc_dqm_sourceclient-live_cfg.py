import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCDQMLIVE")

#-------------------------------------------------
# DQM Module Configuration
#-------------------------------------------------

process.load("DQM.CSCMonitorModule.csc_dqm_sourceclient_cfi")
process.dqmCSCClient.EventProcessor.BINCHECKER_MODE_DDU = cms.untracked.bool(False)
process.dqmCSCClient.EventProcessor.EFF_COLD_SIGFAIL = cms.untracked.double(2.0)
process.dqmCSCClient.EventProcessor.EFF_HOT_THRESHOLD = cms.untracked.double(2.0)
process.dqmCSCClient.EventProcessor.EFF_HOT_SIGFAIL = cms.untracked.double(10.0)
process.dqmCSCClient.EventProcessor.EFF_NODATA_THRESHOLD = cms.untracked.double(0.99)

#process.dqmCSCClient.FractUpdateEventFreq = cms.untracked.uint32(100)
#process.dqmCSCClient.effParameters.threshold_hot = cms.untracked.double(10.0)
#process.dqmCSCClient.effParameters.sigfail_cold = cms.untracked.double(3.0)

#-------------------------------------------------
# Offline DQM Module Configuration
#-------------------------------------------------

process.load("DQMOffline.Muon.CSCMonitor_cfi")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.csc2DRecHits.readBadChambers = cms.bool(False)

#----------------------------
# Event Source
#-----------------------------

process.load("DQM.Integration.test.inputsource_cfi")

process.EventStreamHttpReader.consumerName = 'CSC DQM Consumer'

#----------------------------
# DQM Environment
#-----------------------------

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/csc_reference.root'

#----------------------------
# DQM Playback Environment
#-----------------------------

process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder    = "CSC"

#process.DQM.collectorHost = 'pccmsdqm02.cern.ch'
#process.DQM.collectorHost = 'localhost'
#process.dqmSaver.dirName = '.'

#-----------------------------
# Magnetic Field
#-----------------------------

#process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")

#-------------------------------------------------
# GEOMETRY
#-------------------------------------------------

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-------------------------------------------------
# Global Tag
#-------------------------------------------------

process.load("DQM.Integration.test.FrontierCondition_GT_cfi")

#--------------------------
# Web Service
#--------------------------

process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")
process.AdaptorConfig = cms.Service("AdaptorConfig")

#--------------------------
# Message Logger
#--------------------------

MessageLogger = cms.Service("MessageLogger",

# suppressInfo = cms.untracked.vstring('source'),
  suppressInfo = cms.untracked.vstring('*'),
  cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    WARNING = cms.untracked.PSet(
      limit = cms.untracked.int32(0)
    ),
    noLineBreaks = cms.untracked.bool(False)
  ),
  detailedInfo = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO')
  ),
  critical = cms.untracked.PSet(
    threshold = cms.untracked.string('ERROR')
  ),
  debugModules = cms.untracked.vstring('CSCMonitormodule'),
#  destinations = cms.untracked.vstring('detailedInfo', 
#    'critical', 
#    'cout')

)

#--------------------------
# Sequences
#--------------------------

#process.p = cms.Path(process.dqmCSCClient+process.dqmEnv+process.dqmSaver)
process.p = cms.Path(process.dqmCSCClient * process.muonCSCDigis * process.csc2DRecHits * process.cscSegments * process.cscMonitor + process.dqmEnv + process.dqmSaver)
