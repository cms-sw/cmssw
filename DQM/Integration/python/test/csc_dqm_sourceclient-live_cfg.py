import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCDQMLIVE")

#-------------------------------------------------
# CSC L1 Emulator Configuration
#-------------------------------------------------

process.load("L1Trigger.CSCTriggerPrimitives.CSCTPE_setup_cfi")

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
# for live online DQM in P5
process.load("DQM.Integration.test.inputsource_cfi")

# for testing in lxplus
#process.load("DQM.Integration.test.fileinputsource_cfi")

#----------------------------
# DQM Environment
#-----------------------------

process.load("DQMServices.Components.DQMEnvironment_cfi")

#----------------------------
# DQM Playback Environment
#-----------------------------

process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder    = "CSC"

process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/csc_reference.root'

#process.DQM.collectorHost = 'pccmsdqm02.cern.ch'
#process.DQM.collectorHost = 'localhost'
process.dqmSaver.dirName = '.'

#-----------------------------
# Magnetic Field
#-----------------------------

#process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")

#-------------------------------------------------
# GEOMETRY
#-------------------------------------------------

process.load("Configuration.StandardSequences.Geometry_cff")

#-------------------------------------------------
# Global Tag
#-------------------------------------------------
# Condition for P5 cluster
process.load("DQM.Integration.test.FrontierCondition_GT_cfi")

# Condition for lxplus
#process.load("DQM.Integration.test.FrontierCondition_GT_Offline_cfi") 

#--------------------------
# Service
#--------------------------
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
process.p = cms.Path(process.dqmCSCClient * process.muonCSCDigis * process.cscTriggerPrimitiveDigis * process.lctreader * process.csc2DRecHits * process.cscSegments * process.cscMonitor + process.dqmEnv + process.dqmSaver)

process.castorDigis.InputLabel = cms.InputTag("rawDataCollector")
process.csctfDigis.producer = cms.InputTag("rawDataCollector")
process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataCollector")
process.ecalDigis.InputLabel = cms.InputTag("rawDataCollector")
process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataCollector")
process.gctDigis.inputLabel = cms.InputTag("rawDataCollector")
process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataCollector")
process.gtEvmDigis.EvmGtInputTag = cms.InputTag("rawDataCollector")
process.hcalDigis.InputLabel = cms.InputTag("rawDataCollector")
process.muonCSCDigis.InputObjects = cms.InputTag("rawDataCollector")
process.muonDTDigis.inputLabel = cms.InputTag("rawDataCollector")
process.muonRPCDigis.InputLabel = cms.InputTag("rawDataCollector")
process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataCollector")
process.siPixelDigis.InputLabel = cms.InputTag("rawDataCollector")
process.siStripDigis.ProductLabel = cms.InputTag("rawDataCollector")
process.cscMonitor.FEDRawDataCollectionTag = cms.InputTag("rawDataCollector")
process.dqmCSCClient.InputObjects = cms.untracked.InputTag("rawDataCollector")

#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------

print "Running with run type = ", process.runType.getRunType()

if (process.runType.getRunType() == process.runType.hi_run):
    process.castorDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.csctfDigis.producer = cms.InputTag("rawDataRepacker")
    process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataRepacker")
    process.ecalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataRepacker")
    process.gctDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataRepacker")
    process.gtEvmDigis.EvmGtInputTag = cms.InputTag("rawDataRepacker")
    process.hcalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.muonCSCDigis.InputObjects = cms.InputTag("rawDataRepacker")
    process.muonDTDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.muonRPCDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataRepacker")
    process.siPixelDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.siStripDigis.ProductLabel = cms.InputTag("rawDataRepacker")
    process.cscMonitor.FEDRawDataCollectionTag = cms.InputTag("rawDataRepacker")
    process.dqmCSCClient.InputObjects = cms.untracked.InputTag("rawDataRepacker")
