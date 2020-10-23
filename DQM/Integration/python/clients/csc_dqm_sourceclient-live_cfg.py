from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import sys

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
process.dqmCSCClient.EventProcessor.PREBOOK_ALL_HISTOS = cms.untracked.bool(True)

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

unitTest=False
if 'unitTest=True' in sys.argv:
  unitTest=True

if unitTest:
  process.load("DQM.Integration.config.unittestinputsource_cfi")
  from DQM.Integration.config.unittestinputsource_cfi import options
else:
  # for live online DQM in P5
  process.load("DQM.Integration.config.inputsource_cfi")
  from DQM.Integration.config.inputsource_cfi import options

# for testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")
#from DQM.Integration.config.fileinputsource_cfi import options

#----------------------------
# DQM Environment
#-----------------------------

#----------------------------
# DQM Playback Environment
#-----------------------------

process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder    = "CSC"
process.dqmSaver.tag = "CSC"
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = "CSC"
process.dqmSaverPB.runNumber = options.runNumber


#process.DQM.collectorHost = 'pccmsdqm02.cern.ch'
#process.DQM.collectorHost = 'localhost'

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
# Condition for P5 cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")


#--------------------------------------
# Post-LS1 CSC configuraton parameters
#--------------------------------------
"""Customise digi/reco geometry to use unganged ME1/a channels"""
process.CSCGeometryESModule.useGangedStripsInME1a = False
process.idealForDigiCSCGeometry.useGangedStripsInME1a = False

"""Settings for the upgrade raw vs offline condition channel translation"""
process.CSCIndexerESProducer.AlgoName=cms.string("CSCIndexerPostls1")
process.CSCChannelMapperESProducer.AlgoName=cms.string("CSCChannelMapperPostls1")

process.csc2DRecHits.readBadChannels = cms.bool(False)
process.csc2DRecHits.CSCUseGasGainCorrections = cms.bool(False)


""" Pick up upgrade condions data directly from DB tags using ESPrefer's.
Might be useful when dealing with a global tag that doesn't include
'unganged' CSC conditions.
"""
"""
myconds = [
        ('CSCDBGainsRcd', 'CSCDBGains_ungangedME11A_mc'),
        ('CSCDBNoiseMatrixRcd', 'CSCDBNoiseMatrix_ungangedME11A_mc'),
        ('CSCDBCrosstalkRcd', 'CSCDBCrosstalk_ungangedME11A_mc'),
        ('CSCDBPedestalsRcd', 'CSCDBPedestals_ungangedME11A_mc'),
        ('CSCDBGasGainCorrectionRcd', 'CSCDBGasGainCorrection_ungangedME11A_mc')
        ('CSCDBChipSpeedCorrectionRcd', 'CSCDBChipSpeedCorrection_ungangedME11A_mc')
]

from CalibMuon.Configuration.getCSCConditions_frontier_cff import cscConditions
for (classname, tag) in myconds:
      print classname, tag
      sourcename = 'unganged_' + classname
      process.__setattr__(sourcename, cscConditions.clone())
      process.__getattribute__(sourcename).toGet = cms.VPSet( cms.PSet( record = cms.string(classname), tag = cms.string(tag)) )
      process.__getattribute__(sourcename).connect = cms.string('frontier://FrontierProd/CMS_COND_CSC_000')
      process.__setattr__('esp_' + classname, cms.ESPrefer("PoolDBESSource", sourcename) )

del cscConditions
"""
#---------------------------------------


# Condition for lxplus: change and possibly customise the GT
#from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
#process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run2_data', '')

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

#process.p = cms.Path(process.dqmCSCClient+process.dqmEnv+process.dqmSaver+process.dqmSaverPB)
process.p = cms.Path(process.dqmCSCClient * process.muonCSCDigis * process.csc2DRecHits * process.cscSegments * process.cscMonitor + process.dqmEnv + process.dqmSaver + process.dqmSaverPB)


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
process.siPixelDigis.cpu.InputLabel = cms.InputTag("rawDataCollector")
process.siStripDigis.ProductLabel = cms.InputTag("rawDataCollector")
process.cscMonitor.FEDRawDataCollectionTag = cms.InputTag("rawDataCollector")
process.dqmCSCClient.InputObjects = cms.untracked.InputTag("rawDataCollector")

#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------

print("Running with run type = ", process.runType.getRunType())

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
    process.siPixelDigis.cpu.InputLabel = cms.InputTag("rawDataRepacker")
    process.siStripDigis.ProductLabel = cms.InputTag("rawDataRepacker")
    process.cscMonitor.FEDRawDataCollectionTag = cms.InputTag("rawDataRepacker")
    process.dqmCSCClient.InputObjects = cms.untracked.InputTag("rawDataRepacker")


### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
