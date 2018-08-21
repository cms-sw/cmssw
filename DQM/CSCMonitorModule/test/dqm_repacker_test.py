from __future__ import print_function
import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCDQMLIVE")

#-------------------------------------------------
# DQM Module Configuration
#-------------------------------------------------


process.load("DQM.CSCMonitorModule.test.csc_dqm_sourceclient_dev_cfi")
process.load("DQM.CSCMonitorModule.csc_hlt_dqm_sourceclient_cfi")
#process.load("DQM.CSCMonitorModule.csc_dqm_sourceclient_cfi")
process.dqmCSCClient.EventProcessor.BINCHECKER_MODE_DDU = cms.untracked.bool(False)
process.dqmCSCClient.EventProcessor.EFF_COLD_SIGFAIL = cms.untracked.double(2.0)
process.dqmCSCClient.EventProcessor.EFF_HOT_THRESHOLD = cms.untracked.double(2.0)
process.dqmCSCClient.EventProcessor.EFF_HOT_SIGFAIL = cms.untracked.double(10.0)
process.dqmCSCClient.EventProcessor.EFF_NODATA_THRESHOLD = cms.untracked.double(0.95)
process.dqmCSCClient.EventProcessor.EFF_HOT_SIGFAIL = cms.untracked.double(5.0)

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
# DQM Environment
#-----------------------------

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/csc_reference.root'

#----------------------------
# DQM Playback Environment
#-----------------------------

#process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder    = "CSC"

#process.DQM.collectorHost = 'pccmsdqm02.cern.ch'
#process.DQM.collectorHost = 'localhost'
#process.DQM.collectorPort = 8061
#process.DQM.collectorHost = 'lxplus436.cern.ch'
#process.DQM.collectorPort = 9190
process.dqmSaver.convention = 'Online'
process.dqmSaver.producer = 'DQM'
#process.dqmSaver.workflow = '/Test/CSC/DQM'
process.dqmSaver.dirName = '/tmp/barvic/dqmdata'
process.DQMStore.verbose =  cms.untracked.int32(1)
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

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

#process.load("DQM.Integration.test.FrontierCondition_GT_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'GR_R_52_V7::All'
process.GlobalTag.globaltag = 'GR_E_V37::All'



"""Customise digi/reco geometry to use unganged ME1/a channels"""
process.CSCGeometryESModule.useGangedStripsInME1a = False
process.idealForDigiCSCGeometry.useGangedStripsInME1a = False

"""Settings for the upgrade raw vs offline condition channel translation"""
process.CSCIndexerESProducer.AlgoName=cms.string("CSCIndexerPostls1")
process.CSCChannelMapperESProducer.AlgoName=cms.string("CSCChannelMapperPostls1")

process.csc2DRecHits.readBadChannels = cms.bool(False)
process.csc2DRecHits.CSCUseGasGainCorrection = cms.bool(False)


""" Pick up upgrade condions data directly from DB tags using ESPrefer's.
Might be useful when dealing with a global tag that doesn't include
'unganged' CSC conditions.
"""
myconds = [
        ('CSCDBGainsRcd', 'CSCDBGains_ungangedME11A_mc'),
        ('CSCDBNoiseMatrixRcd', 'CSCDBNoiseMatrix_ungangedME11A_mc'),
        ('CSCDBCrosstalkRcd', 'CSCDBCrosstalk_ungangedME11A_mc'),
        ('CSCDBPedestalsRcd', 'CSCDBPedestals_ungangedME11A_mc'),
        ('CSCDBGasGainCorrectionRcd', 'CSCDBGasGainCorrection_ungangedME11A_mc'),
        ('CSCDBChipSpeedCorrectionRcd', 'CSCDBChipSpeedCorrection_ungangedME11A_mc')
]

from CalibMuon.Configuration.getCSCConditions_frontier_cff import cscConditions
for (classname, tag) in myconds:
      print(classname, tag)
      sourcename = 'unganged_' + classname
      process.__setattr__(sourcename, cscConditions.clone())
      process.__getattribute__(sourcename).toGet = cms.VPSet( cms.PSet( record = cms.string(classname), tag = cms.string(tag)) )
      process.__getattribute__(sourcename).connect = cms.string('frontier://FrontierProd/CMS_COND_CSC_000')
      process.__setattr__('esp_' + classname, cms.ESPrefer("PoolDBESSource", sourcename) )

del cscConditions

#--------------------------
# Web Service
#--------------------------

#process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")
#process.AdaptorConfig = cms.Service("AdaptorConfig")

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

#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------

#print "Running with run type = ", process.runType.getRunType()

#if (process.runType.getRunType() == process.runType.hi_run):
#    process.castorDigis.InputLabel = cms.InputTag("rawDataRepacker")
#    process.csctfDigis.producer = cms.InputTag("rawDataRepacker")
#    process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataRepacker")
#    process.ecalDigis.InputLabel = cms.InputTag("rawDataRepacker")
#    process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataRepacker")
#    process.gctDigis.inputLabel = cms.InputTag("rawDataRepacker")
#    process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataRepacker")
#    process.gtEvmDigis.EvmGtInputTag = cms.InputTag("rawDataRepacker")
#    process.hcalDigis.InputLabel = cms.InputTag("rawDataRepacker")
#    process.muonCSCDigis.InputObjects = cms.InputTag("rawDataRepacker")
#    process.muonDTDigis.inputLabel = cms.InputTag("rawDataRepacker")
#    process.muonRPCDigis.InputLabel = cms.InputTag("rawDataRepacker")
#    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataRepacker")
#    process.siPixelDigis.InputLabel = cms.InputTag("rawDataRepacker")
#    process.siStripDigis.ProductLabel = cms.InputTag("rawDataRepacker")
#    process.cscMonitor.FEDRawDataCollectionTag = cms.InputTag("rawDataRepacker")
#    process.dqmCSCClient.InputObjects = cms.untracked.InputTag("rawDataRepacker")


process.cscpacker = cms.EDProducer("CSCDigiToRawModule",
    wireDigiTag = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
    stripDigiTag = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
    comparatorDigiTag = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi"),
    alctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCALCTDigi"),
    clctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCCLCTDigi"),
    preTriggerTag = cms.InputTag("simCscTriggerPrimitiveDigis"),
    correlatedLCTDigiTag = cms.InputTag("muonCSCDigis","MuonCSCCorrelatedLCTDigi"),
    alctWindowMin = cms.int32(-3),
    alctWindowMax = cms.int32(3),
    clctWindowMin = cms.int32(-3),
    clctWindowMax = cms.int32(3),
    preTriggerWindowMin = cms.int32(-3),
    preTriggerWindowMax = cms.int32(1)
)

process.cscpacker.usePreTriggers = cms.untracked.bool(False)
process.cscpacker.useFormatVersion = cms.untracked.uint32(2013)

#----------------------------
# Event Source
#-----------------------------

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    #debugFlag = cms.untracked.bool(True),
    #debugVebosity = cms.untracked.uint32(10),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
    fileNames = cms.untracked.vstring(
#	'file:/tmp/barvic/csc_00221766_Cosmic.root'
        'file:/tmp/barvic/digi_test.root'
#	'file:/tmp/barvic/digi_packer_test.root'
#	'rfio:/castor/cern.ch/cms/store/data/Commissioning2014/Cosmics/RAW/v1/000/220/744/00000/0C7ECA47-C4BE-E311-BDAB-02163E00E734.root'
#       	'rfio:/castor/cern.ch/cms/store/data/Run2012A/SingleMu/RAW/v1/000/193/621/00FED4BE-8D98-E111-B366-001D09F2924F.root',
#	'rfio:/castor/cern.ch/cms/store/data/Run2012A/SingleMu/RAW/v1/000/193/621/042AC95C-8F98-E111-B665-0025901D631E.root'
# ==
    )
)

# Switch to packer FED RAW data collection tag "cscpacker", "CSCRawData"
process.muonCSCDigis.InputObjects = cms.InputTag("cscpacker", "CSCRawData")
process.dqmCSCClient.InputObjects = cms.untracked.InputTag("cscpacker", "CSCRawData")
process.cscMonitor.FEDRawDataCollectionTag = cms.InputTag("cscpacker", "CSCRawData")


#--------------------------
# Sequences
#--------------------------

process.p = cms.Path( process.cscpacker * process.dqmCSCClient * process.muonCSCDigis * process.csc2DRecHits * process.cscSegments * process.cscMonitor + process.dqmEnv + process.dqmSaver)
#process.p = cms.Path( process.cscpacker * process.dqmCSCClient + process.dqmEnv + process.dqmSaver)

#process.dqmsave_step = cms.Path(process.dqmSaver)

# Schedule definition
#process.schedule = cms.Schedule(
#                                process.dqmCSCClient,
#                                process.muonCSCDigis,
#                                process.dqmEnv,
#                                process.dqmsave_step
#    )
