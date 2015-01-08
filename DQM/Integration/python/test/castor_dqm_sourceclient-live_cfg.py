import FWCore.ParameterSet.Config as cms

process = cms.Process("CASTORDQM")
#=================================
# Event Source
#================================+
# for live online DQM in P5
process.load("DQM.Integration.test.inputsource_cfi")

# for testing in lxplus
#process.load("DQM.Integration.test.fileinputsource_cfi")

#================================
# DQM Environment
#================================
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
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CastorDbProducer = cms.ESProducer("CastorDbProducer")

process.es_pool = cms.ESSource(
   "PoolDBESSource",
   process.CondDBSetup,
   timetype = cms.string('runnumber'),
   connect = cms.string('frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOn Prod)(retrieve-ziplevel=0)(failovertoserver=no)/CMS_COND_31X_HCAL'),
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
process.load("EventFilter.CastorRawToDigi.CastorRawToDigi_cfi")
process.load("RecoLocalCalo.CastorReco.CastorSimpleReconstructor_cfi")

process.castorreco.tsFromDB = cms.untracked.bool(False)

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
		 
			   # Define Labels
                           digiLabel            = cms.InputTag("castorDigis"),
                           rawLabel             = cms.InputTag("source"),
                           unpackerReportLabel  = cms.InputTag("castorDigis"),
                           CastorRecHitLabel    = cms.InputTag("castorreco"),
                           CastorTowerLabel     = cms.InputTag("CastorTowerReco"),
                           CastorBasicJetsLabel = cms.InputTag("ak7CastorJets"),
                           CastorJetIDLabel     = cms.InputTag("ak7CastorJetID"),
                          
			   DataIntMonitor= cms.untracked.bool(True),
			   TowerJetMonitor= cms.untracked.bool(False),

                           DigiMonitor = cms.untracked.bool(True),
                           DigiPerChannel = cms.untracked.bool(True), 
                           DigiInFC = cms.untracked.bool(False),
                          
                           RecHitMonitor = cms.untracked.bool(True), 
			   RecHitsPerChannel = cms.untracked.bool(True),

                           ChannelQualityMonitor= cms.untracked.bool(True),
                           nThreshold = cms.untracked.double(60),
                           dThreshold = cms.untracked.double(1.0),
                           OfflineMode = cms.untracked.bool(False),
                           averageEnergyMethod = cms.untracked.bool(True),          

                           PSMonitor= cms.untracked.bool(True),
                           numberSigma = cms.untracked.double(1.5),
                           thirdRegionThreshold =  cms.untracked.double(999999.0), # to avoid empty cells in reportSummary            
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
#process.dqmSaver.producer = 'DQM'
#process.dqmSaver.dirName = '/tmp/'
#process.dqmSaver.convention = 'Online'
#process.dqmSaver.saveByRun = True
# for local test
process.dqmSaver.dirName = '.'

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


process.castorDigis.InputLabel = cms.InputTag("rawDataCollector")
process.castorMonitor.rawLabel = cms.InputTag("rawDataCollector")
    
#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------

print "Running with run type = ", process.runType.getRunType()

if (process.runType.getRunType() == process.runType.hi_run):
    process.castorDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.castorMonitor.rawLabel = cms.InputTag("rawDataRepacker")
