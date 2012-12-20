import FWCore.ParameterSet.Config as cms

process = cms.Process("CASTORDQM")
#=================================
# Event Source
#================================+

### to use a dat file
#process.source = cms.Source("NewEventStreamFileReader",
#    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/c/campbell/scratch0/first100M_MWGR_41.00116713.0001.A.storageManager.00.0000.dat')
# )

### to use a root file


process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    'root://eoscms//eos/cms/store/data/Run2011A/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v6/0000/182AF1D2-41CB-E011-8661-002354EF3BE0.root',
    'root://eoscms//eos/cms/store/data/Run2011A/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v6/0000/1E7A9033-0EC2-E011-9191-001A928116F4.root',
    'root://eoscms//eos/cms/store/data/Run2011A/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v6/0000/3284739C-0DC2-E011-966C-002618943842.root',
    'root://eoscms//eos/cms/store/data/Run2011A/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v6/0000/3E80B615-3BCB-E011-B288-0018F3D09670.root',
    'root://eoscms//eos/cms/store/data/Run2011A/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v6/0000/441707B7-42CB-E011-A2D3-00261894384A.root',
    'root://eoscms//eos/cms/store/data/Run2011A/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v6/0000/468F37BA-F8C1-E011-B2D1-0018F3D09706.root'),
                            )



process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)


#================================
# DQM Environment
#================================
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
#process.DQMStore.referenceFileName = 'castor_reference.root'

#depreciated use the next block
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.Reconstruction_Data_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

#process.load("DQM.Integration.test.environment_cfi")

from DQMServices.Core.DQM_cfg import *


DQM.collectorHost = 'dqm-prod-local.cms'
DQM.collectorPort = 9090

from DQMServices.Components.DQMEnvironment_cfi import *

process.dqmEnv.subSystemFolder = "Castor"

process.load("FWCore.MessageLogger.MessageLogger_cfi")

#============================================
# Castor Conditions: from Global Conditions Tag 
#============================================

#get from global tag
#from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag.globaltag = 'GR_R_52_V10::All'


#get explicit from db
process.load("CondCore.DBCommon.CondDBSetup_cfi")
#process.castor_db_producer = cms.ESProducer("CastorDbProducer") 

#process.es_pool = cms.ESSource(
#   "PoolDBESSource",
#   process.CondDBSetup,
#   timetype = cms.string('runnumber'),
   # connect = cms.string('frontier://cmsfrontier.cern.ch:8000/FrontierPrep/CMS_COND_30X_HCAL'),
#   connect = cms.string('frontier://cmsfrontier.cern.ch:8000/FrontierProd/CMS_COND_31X_HCAL'),
#   authenticationMethod = cms.untracked.uint32(0),
#   toGet = cms.VPSet(
#       cms.PSet(
#           record = cms.string('CastorPedestalsRcd'),
#           tag = cms.string('castor_pedestals_v1.0')
#           ),
#       cms.PSet(
#          record = cms.string('CastorPedestalWidthsRcd'),
#           tag = cms.string('castor_pedestalwidths_v1.0')
#           ),
#       cms.PSet(
#           record = cms.string('CastorGainsRcd'),
#           tag = cms.string('castor_gains_v1.0')
#           ),
#       cms.PSet(
#           record = cms.string('CastorGainWidthsRcd'),
#           tag = cms.string('castor_gainwidths_v1.0')
#           ),
#       cms.PSet(
#           record = cms.string('CastorQIEDataRcd'),
#           tag = cms.string('castor_qie_v1.0')
#           ),
#       cms.PSet(
#           record = cms.string('CastorChannelQualityRcd'),
#           tag = cms.string('castor_channelquality_v1.0')
#           ),
#       cms.PSet(
#           record = cms.string('CastorElectronicsMapRcd'),
#           tag = cms.string('castor_emap_dcc_v1.0')
#           )
#   )
#)



#-----------------------------
# Castor DQM Source + SimpleReconstrctor
#-----------------------------
#process.load("DQM.CastorMonitor.CastorMonitorModule_cfi")
process.load("EventFilter.CastorRawToDigi.CastorRawToDigi_cfi")
process.load("RecoLocalCalo.CastorReco.CastorSimpleReconstructor_cfi")

process.castorreco.tsFromDB = cms.untracked.bool(False)

process.castorDigis = cms.EDProducer("CastorRawToDigi",
    # Optional filter to remove any digi with "data valid" off, "error" on, 
    # or capids not rotating
    FilterDataQuality = cms.bool(True),
    # Number of the first CASTOR FED.  If this is not specified, the
    # default from FEDNumbering is used.
    CastorFirstFED = cms.untracked.int32(690),
    # FED numbers to unpack.  If this is not specified, all FEDs from
    # FEDNumbering will be unpacked.
    FEDs = cms.untracked.vint32( 690, 691, 692 ),
    # Do not complain about missing FEDs
    ExceptionEmptyData = cms.untracked.bool(False),
    # Do not complain about missing FEDs
    ComplainEmptyData = cms.untracked.bool(False),
    # At most ten samples can be put into a digi, if there are more
    # than ten, firstSample and lastSample select which samples
    # will be copied to the digi
    firstSample = cms.int32(0),
    lastSample = cms.int32(9),
    # castor technical trigger processor
    UnpackTTP = cms.untracked.bool(True),
    # report errors
    silent = cms.untracked.bool(False),
    #
    InputLabel = cms.InputTag("rawDataCollector")
)

process.castorMonitor = cms.EDAnalyzer("CastorMonitorModule",
                           ### GLOBAL VARIABLES
                           debug = cms.untracked.int32(0), # make debug an int so that different
                                                           # values can trigger different levels of messaging
                           # Turn on/off timing diagnostic info
                           showTiming          = cms.untracked.bool(False),
                           dump2database       = cms.untracked.bool(False),
                           pedestalsInFC = cms.untracked.bool(False),
			   # Define Labels
                           digiLabel            = cms.InputTag("castorDigis"),
                           rawLabel             = cms.InputTag("rawDataCollector"),
                           unpackerReportLabel  = cms.InputTag("castorDigis"),
                           CastorRecHitLabel    = cms.InputTag("castorreco"),
                           CastorTowerLabel     = cms.InputTag("CastorTowerReco"),
                           CastorBasicJetsLabel = cms.InputTag("ak7BasicJets"),
                           CastorJetIDLabel     = cms.InputTag("ak7CastorJetID"),
                          
			   DataIntMonitor= cms.untracked.bool(True),
			   TowerJetMonitor= cms.untracked.bool(True),

                           DigiMonitor = cms.untracked.bool(True),
                           DigiPerChannel = cms.untracked.bool(True), 
                           DigiInFC = cms.untracked.bool(False),
                          
                           RecHitMonitor = cms.untracked.bool(True), 
			   RecHitsPerChannel = cms.untracked.bool(True),

                           ChannelQualityMonitor= cms.untracked.bool(True),
                           nThreshold = cms.untracked.double(60),
                           dThreshold = cms.untracked.double(0.1),
                           OfflineMode = cms.untracked.bool(True),
                           averageEnergyMethod = cms.untracked.bool(True),
                                     
                           PSMonitor= cms.untracked.bool(True),
                           numberSigma = cms.untracked.double(1.5),
                           thirdRegionThreshold =  cms.untracked.double(999999.0),        
                           EDMonitor= cms.untracked.bool(False),
                           HIMonitor= cms.untracked.bool(True),
                                       
                           diagnosticPrescaleTime = cms.untracked.int32(-1),
                           diagnosticPrescaleUpdate = cms.untracked.int32(-1),
                           diagnosticPrescaleLS = cms.untracked.int32(-1),
                             
                           LEDMonitor = cms.untracked.bool(True),
                           LEDPerChannel = cms.untracked.bool(True),
                           FirstSignalBin = cms.untracked.int32(0),
                           LastSignalBin = cms.untracked.int32(9),
                           LED_ADC_Thresh = cms.untracked.double(-1000.0)      
                           )

### the filename prefix 
process.dqmSaver.producer = 'DQM'
process.dqmSaver.dirName = '.'
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = True

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

#process.p = cms.Path(process.castorMonitor*process.dqmEnv*process.dqmSaver)
process.p = cms.Path(process.castorDigis*process.castorreco*process.castorMonitor*process.dqmEnv*process.dqmSaver)
#process.p = cms.Path(process.castorMonitor*process.dqmEnv*process.dqmSaver)
#process.p = cms.Path(process.castorMonitor*process.dqmEnv*process.dqmSaver)

