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
   # fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/121/993/D04EA868-5FD6-DE11-B372-003048D2BE08.root')
   #fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RAW/v3/000/129/468/8A1BC712-8A24-DF11-800B-000423D9970C.root')
   fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/data/Run2010A/MinimumBias/RAW/v1/000/144/114/9AB9A0B1-F1B3-DF11-BCBF-001D09F24FEC.root')

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

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder = "Castor"

process.load("FWCore.MessageLogger.MessageLogger_cfi")

#============================================
# Castor Conditions: from Global Conditions Tag 
#============================================

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.castor_db_producer = cms.ESProducer("CastorDbProducer") 

process.es_pool = cms.ESSource(
   "PoolDBESSource",
   process.CondDBSetup,
   timetype = cms.string('runnumber'),
   # connect = cms.string('frontier://cmsfrontier.cern.ch:8000/FrontierPrep/CMS_COND_30X_HCAL'),
   connect = cms.string('frontier://cmsfrontier.cern.ch:8000/FrontierProd/CMS_COND_31X_HCAL'),
   authenticationMethod = cms.untracked.uint32(0),
   toGet = cms.VPSet(
       cms.PSet(
           record = cms.string('CastorPedestalsRcd'),
           tag = cms.string('castor_pedestals_v1.0')
           ),
       cms.PSet(
           record = cms.string('CastorPedestalWidthsRcd'),
           tag = cms.string('castor_pedestalwidths_v1.0')
           ),
       cms.PSet(
           record = cms.string('CastorGainsRcd'),
           tag = cms.string('castor_gains_v1.0')
           ),
       cms.PSet(
           record = cms.string('CastorGainWidthsRcd'),
           tag = cms.string('castor_gainwidths_v1.0')
           ),
       cms.PSet(
           record = cms.string('CastorQIEDataRcd'),
           tag = cms.string('castor_qie_v1.0')
           ),
       cms.PSet(
           record = cms.string('CastorChannelQualityRcd'),
           tag = cms.string('castor_channelquality_v1.0')
           ),
       cms.PSet(
           record = cms.string('CastorElectronicsMapRcd'),
           tag = cms.string('castor_emap_dcc_v1.0')
           )
   )
)




#-----------------------------
# Castor DQM Source + SimpleReconstrctor
#-----------------------------
#process.load("DQM.CastorMonitor.CastorMonitorModule_cfi")
process.load("EventFilter.CastorRawToDigi.CastorRawToDigi_cfi")
process.load("RecoLocalCalo.CastorReco.CastorSimpleReconstructor_cfi")

process.castorDigis = cms.EDProducer("CastorRawToDigi",
   CastorFirstFED = cms.untracked.int32(690),
   FilterDataQuality = cms.bool(True),
   ExceptionEmptyData = cms.untracked.bool(True),
   InputLabel = cms.InputTag("source"),
   UnpackCalib = cms.untracked.bool(False),
   FEDs = cms.untracked.vint32(690,691,692),
   lastSample = cms.int32(9),
   firstSample = cms.int32(0)
) 

process.castorMonitor = cms.EDAnalyzer("CastorMonitorModule",
                           ### GLOBAL VARIABLES
                           debug = cms.untracked.int32(0), # make debug an int so that different
                                                           # values can trigger different levels of messaging
                           # Turn on/off timing diagnostic info
                           showTiming          = cms.untracked.bool(False),
                           dump2database       = cms.untracked.bool(False),
                           pedestalsInFC = cms.untracked.bool(False),
                           digiLabel = cms.InputTag("castorDigis"),
                           rawLabel = cms.InputTag("castorRaw"),
                           CastorRecHitLabel = cms.InputTag("castorreco"),
                          
                                       
                           DigiMonitor = cms.untracked.bool(True),
                           DigiPerChannel = cms.untracked.bool(True), 
                           DigiInFC = cms.untracked.bool(False),
                          
                           RecHitMonitor = cms.untracked.bool(True), 
			   RecHitsPerChannel = cms.untracked.bool(True),

                           ChannelQualityMonitor= cms.untracked.bool(True),
                           nThreshold = cms.untracked.double(70),
                           dThreshold = cms.untracked.double(0.1),
                           OfflineMode = cms.untracked.bool(True),
                           averageEnergyMethod = cms.untracked.bool(True),
                                     
                           PSMonitor= cms.untracked.bool(True),
                           numberSigma = cms.untracked.double(1.5),
                           thirdRegionThreshold =  cms.untracked.double(100),        
                           EDMonitor= cms.untracked.bool(True),
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

process.p = cms.Path(process.castorDigis*process.castorreco*process.castorMonitor*process.dqmEnv*process.dqmSaver)
#process.p = cms.Path(process.castorDigis*process.castorMonitor*process.dqmEnv*process.dqmSaver)
#process.p = cms.Path(process.castorMonitor*process.dqmEnv*process.dqmSaver)

