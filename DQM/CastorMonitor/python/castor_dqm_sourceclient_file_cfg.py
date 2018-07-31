import FWCore.ParameterSet.Config as cms

process = cms.Process("CASTORDQM")
#=================================
# Event Source
#================================+


### data root file
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
#'root://eoscms//eos/cms/store/hidata/HIRun2013/PAMinBiasUPC/RAW/v1/000/210/885/00000/0AC926B5-7268-E211-91F3-BCAEC5329708.root'),
#'file:/tmp/popov/1257F374-8BB0-E611-A096-02163E0145DD-f10.root'),
#'file:/afs/cern.ch/user/p/popov/scratch_bk/data/4C7A69AE-6692-E511-A446-02163E0119EB-RAW262272.root'),

#'file:/afs/cern.ch/user/p/popov/scratch_bk/data/64E2F5E1-5892-E511-B421-02163E0137E8-RAW262270.root'),
#'file:/eos/user/p/popov/HI2015/AOD-665ED244-96AB-E511-9A9E-02163E011E5B.root'),

#'file:/eos/cms/store/data/Run2018C/ZeroBias/RAW/v1/000/320/260/00000/80A7FA82-AD90-E811-B3B3-FA163EE997B7.root'),
#'file:/eos/cms/store/data/Run2018C/ZeroBias/RAW/v1/000/320/285/00000/6C275008-4490-E811-9AB4-FA163E7FC1F6.root'),
#'file:/eos/cms/store/data/Run2018C/MinimumBias/RAW/v1/000/320/317/00000/3AA90BCE-6390-E811-8093-FA163E3A93BC.root'),
#'file:/eos/cms/store/data/Run2018C/ZeroBias/RAW/v1/000/320/317/00000/525D2460-6A90-E811-AF46-FA163E9626F3.root'),
'file:/eos/user/p/popov/rundata/Castor2018/525D2460-6A90-E811-RAWrun320317.root'),
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

#================================
# DQM framework
#================================
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "Castor"
process.dqmEnv.eventInfoFolder = "EventInfo"
#process.dqmSaver.producer = 'DQM'
process.dqmSaver.path = ""
process.dqmSaver.tag = "Castor"

process.load("FWCore.MessageLogger.MessageLogger_cfi")

#============================================
# Castor Conditions: from Global Conditions Tag 
#============================================

#get from global tag
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag.globaltag = 'GR_R_61_V7::All' #autoCond['run2_data']
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run2_data']

#get explicit from db
#process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("CondCore.CondDB.CondDB_cfi")

#process.castor_db_producer = cms.ESProducer("CastorDbProducer") 

#-----------------------------
# Castor DQM Source + SimpleReconstrctor
#-----------------------------
#process.load("DQM.CastorMonitor.CastorMonitorModule_cfi")
process.load("RecoLocalCalo.CastorReco.CastorSimpleReconstructor_cfi")

process.castorreco.tsFromDB = cms.bool(False)

#process.load("EventFilter.CastorRawToDigi.CastorRawToDigi_cfi")
from EventFilter.CastorRawToDigi.CastorRawToDigi_cff import *
#process.castorDigis = castorDigis.clone( UnpackZDC = cms.bool(False))
process.castorDigis = castorDigis.clone()

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
process.castorMonitor = DQMEDAnalyzer("CastorMonitorModule",
                           ### GLOBAL VARIABLES
   debug = cms.untracked.int32(1), #(=0 - no messages)
                           # Turn on/off timing diagnostic
                           showTiming          = cms.untracked.bool(False),
#                      dump2database       = cms.untracked.bool(False),
#                      pedestalsInFC = cms.untracked.bool(False),

			   # Define Labels
     l1tStage2uGtSource = cms.InputTag("gtStage2Digis"),
#     tagTriggerResults   = cms.InputTag("TriggerResults"),
     tagTriggerResults   = cms.InputTag('TriggerResults','','HLT'),
    HltPaths  = cms.vstring("HLT_ZeroBias","HLT_Random"),

                           digiLabel            = cms.InputTag("castorDigis"),
                           rawLabel 		= cms.InputTag("rawDataCollector"),
                           unpackerReportLabel  = cms.InputTag("castorDigis"),
                           CastorRecHitLabel    = cms.InputTag("castorreco"),
                           CastorTowerLabel     = cms.InputTag("CastorTowerReco"),
                           CastorBasicJetsLabel = cms.InputTag("ak7CastorJets"),
                           CastorJetIDLabel     = cms.InputTag("ak7CastorJetID"),
                                                    
			   DataIntMonitor= cms.untracked.bool(True),
			   TowerJetMonitor= cms.untracked.bool(True),

                           DigiMonitor = cms.untracked.bool(True),
                          
                           RecHitMonitor = cms.untracked.bool(True), 

                             
#                           LEDMonitor = cms.untracked.bool(True),
#                           LEDPerChannel = cms.untracked.bool(True),
                           FirstSignalBin = cms.untracked.int32(0),
                           LastSignalBin = cms.untracked.int32(9)
)

### the filename prefix 
#process.dqmSaver.dirName = '.'
#convention does not already exist# process.dqmSaver.convention = 'Online'
#saveByRun does not already exist# process.dqmSaver.saveByRun = True

#-----------------------------
# Scheduling
#-----------------------------
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound', 
        'TooManyProducts', 
        'TooFewProducts')
)

# castorDigis   -> CastorRawToDigi_cfi
# castorreco    -> CastorSimpleReconstructor_cfi
# castorMonitor -> CastorMonitorModule_cfi
process.out = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string('my.root')
)

#process.p = cms.Path(process.castorMonitor*process.dqmEnv*process.dqmSaver)
process.p = cms.Path(process.castorDigis*process.castorreco*process.castorMonitor)
#process.p = cms.Path(process.castorMonitor)

process.end_path = cms.EndPath(
  process.dqmEnv +
  process.dqmSaver
)

process.schedule = cms.Schedule(
  process.p,
  process.end_path
)

