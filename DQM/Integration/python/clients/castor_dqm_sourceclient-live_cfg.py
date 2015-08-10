import FWCore.ParameterSet.Config as cms

process = cms.Process("CASTORDQM")
#=================================
# Event Source
#================================+
# for live online DQM in P5
process.load("DQM.Integration.config.inputsource_cfi")

# for testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")

#================================
# DQM Environment
#================================
#process.DQMStore.referenceFileName = 'castor_reference.root'

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "Castor"
process.dqmSaver.tag = "Castor"

process.load("FWCore.MessageLogger.MessageLogger_cfi")


#============================================
# Castor Conditions: from Global Conditions Tag 
#============================================
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CastorDbProducer = cms.ESProducer("CastorDbProducer")

###process.es_pool = cms.ESSource(
###   "PoolDBESSource",
###   process.CondDBSetup,
###   timetype = cms.string('runnumber'),
###   connect = cms.string('frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOn Prod)(retrieve-ziplevel=0)(failovertoserver=no)/CMS_COND_31X_HCAL'),
###   authenticationMethod = cms.untracked.uint32(0),
###   toGet = cms.VPSet(
###       cms.PSet(
###           record = cms.string('CastorPedestalsRcd'),
###           tag = cms.string('castor_pedestals_v1.0')
###           ),
###       cms.PSet(
###           record = cms.string('CastorPedestalWidthsRcd'),
###           tag = cms.string('castor_pedestalwidths_v1.0')
###           ),
###       cms.PSet(
###           record = cms.string('CastorGainsRcd'),
###           tag = cms.string('castor_gains_v1.0')
###           ),
###       cms.PSet(
###           record = cms.string('CastorGainWidthsRcd'),
###           tag = cms.string('castor_gainwidths_v1.0')
###           ),
###       cms.PSet(
###           record = cms.string('CastorQIEDataRcd'),
###           tag = cms.string('castor_qie_v1.0')
###           ),
###       cms.PSet(
###           record = cms.string('CastorChannelQualityRcd'),
###           tag = cms.string('castor_channelquality_v1.0')
###           ),
###       cms.PSet(
###           record = cms.string('CastorElectronicsMapRcd'),
###           tag = cms.string('castor_emap_dcc_v1.0')
###           )
###   )
###)

process.es_pool0 = cms.ESSource(
   "PoolDBESSource",
   process.CondDBSetup,
   timetype = cms.string('runnumber'),
   ###connect = cms.string('frontier://FrontierProd/CMS_COND_31X_HCAL'),
   connect = cms.string('frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOn Prod)(retrieve-ziplevel=0)(failovertoserver=no)/CMS_COND_31X_HCAL'),
   authenticationMethod = cms.untracked.uint32(0),
   toGet = cms.VPSet(
       cms.PSet(
           record = cms.string('CastorGainsRcd'),
           tag = cms.string('CastorGains_v2.1_hlt') #all gains=1, corresponding to GR_P_V43D and CastorGains_v2.1_hlt in frontier://PromptProd/CMS_COND_31X_HCAL
           )
   )
)

process.es_pool = cms.ESSource(
   "PoolDBESSource",
   process.CondDBSetup,
   timetype = cms.string('runnumber'),
   #connect = cms.string('frontier://cmsfrontier.cern.ch:8000/FrontierPrep/CMS_COND_30X_HCAL'),
   #connect = cms.string('frontier://cmsfrontier.cern.ch:8000/FrontierProd/CMS_COND_31X_HCAL'),
   ###connect = cms.string('frontier://FrontierProd/CMS_COND_HCAL_000'),
   connect = cms.string('frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOn Prod)(retrieve-ziplevel=0)(failovertoserver=no)/CMS_COND_HCAL_000'),
   authenticationMethod = cms.untracked.uint32(0),
   toGet = cms.VPSet(
       cms.PSet(
           record = cms.string('CastorSaturationCorrsRcd'),
           tag = cms.string('CastorSaturationCorrs_v3.00_offline')
           ),
#
       cms.PSet(
           record = cms.string('CastorPedestalsRcd'),
           tag = cms.string('CastorPedestals_v3.00_offline')
           ),
#
       cms.PSet(
           record = cms.string('CastorPedestalWidthsRcd'),
           tag = cms.string('CastorPedestalWidths_v3.00_offline')
           ),
#
#       cms.PSet(
#           record = cms.string('CastorGainsRcd'),
#           tag = cms.string('CastorGains_v3.00_offline')
#           ),
#
       cms.PSet(
           record = cms.string('CastorGainWidthsRcd'),
           tag = cms.string('CastorGainWidths_v3.00_offline')
           ),
       cms.PSet(
           record = cms.string('CastorQIEDataRcd'),
           tag = cms.string('CastorQIEData_v3.00_offline')
           ),
       cms.PSet(
           record = cms.string('CastorChannelQualityRcd'),
           tag = cms.string('CastorChannelQuality_v3.00_offline')
           ),
       cms.PSet(
           record = cms.string('CastorElectronicsMapRcd'),
           tag = cms.string('CastorElectronicsMap_v3.00_offline')
           )
   )
)


#-----------------------------
# Castor DQM Source + SimpleReconstrctor
#-----------------------------
###process.load("EventFilter.CastorRawToDigi.CastorRawToDigi_cfi")
###process.load("RecoLocalCalo.CastorReco.CastorSimpleReconstructor_cfi")
process.castorreco = cms.EDProducer("CastorSimpleReconstructor",
                                    correctionPhaseNS = cms.double(0.0),
                                    digiLabel = cms.InputTag("castorDigis"),
                                    samplesToAdd = cms.int32(10),
                                    Subdetector = cms.string('CASTOR'),
                                    firstSample = cms.int32(0),
                                    correctForPhaseContainment = cms.bool(False),
                                    correctForTimeslew = cms.bool(False),
                                    tsFromDB = cms.bool(False), #True
                                    setSaturationFlag = cms.bool(True),
                                    maxADCvalue = cms.int32(127),
                                    doSaturationCorr = cms.bool(False) #True
)
###process.castorreco.tsFromDB = cms.untracked.bool(False)
process.load('RecoLocalCalo.Castor.Castor_cff') #castor tower and jet reconstruction

###process.castorDigis = cms.EDProducer("CastorRawToDigi",
###   CastorFirstFED = cms.untracked.int32(690),
###   FilterDataQuality = cms.bool(True),
###   ExceptionEmptyData = cms.untracked.bool(True),
###   InputLabel = cms.InputTag("source"),
###   UnpackCalib = cms.untracked.bool(False),
###   FEDs = cms.untracked.vint32(690,691,692),
###   lastSample = cms.int32(9),
###   firstSample = cms.int32(0)
###) 

process.castorDigis = cms.EDProducer("CastorRawToDigi",
   CastorFirstFED = cms.int32(690),
   FilterDataQuality = cms.bool(True),
   ExceptionEmptyData = cms.untracked.bool(True),
   InputLabel = cms.InputTag("rawDataCollector"), #source #rawDataCollector
   #UnpackCalib = cms.untracked.bool(False),
   # castor technical trigger processor
   UnpackTTP = cms.bool(True),
   FEDs = cms.untracked.vint32(690,691,692),
   lastSample = cms.int32(9),
   firstSample = cms.int32(0),
   CastorCtdc = cms.bool(False),
   UseNominalOrbitMessageTime = cms.bool(True),
   ExpectedOrbitMessageTime = cms.int32(-1)#,
   # Do not complain about missing FEDs
   #ComplainEmptyData = cms.untracked.bool(False),
)

###process.castorMonitor = cms.EDAnalyzer("CastorMonitorModule",
###                           ### GLOBAL VARIABLES
###                           debug = cms.untracked.int32(0), # make debug an int so that different
###                                                           # values can trigger different levels of messaging
###                           # Turn on/off timing diagnostic info
###                           showTiming          = cms.untracked.bool(False),
###                           dump2database       = cms.untracked.bool(False),
###                           pedestalsInFC = cms.untracked.bool(False),			   
###		 
###			   # Define Labels
###                           digiLabel            = cms.InputTag("castorDigis"),
###                           rawLabel             = cms.InputTag("source"),
###                           unpackerReportLabel  = cms.InputTag("castorDigis"),
###                           CastorRecHitLabel    = cms.InputTag("castorreco"),
###                           CastorTowerLabel     = cms.InputTag("CastorTowerReco"),
###                           CastorBasicJetsLabel = cms.InputTag("ak7BasicJets"),
###                           CastorJetIDLabel     = cms.InputTag("ak7CastorJetID"),
###                          
###			   DataIntMonitor= cms.untracked.bool(True),
###			   TowerJetMonitor= cms.untracked.bool(False),
###
###                           DigiMonitor = cms.untracked.bool(True),
###                           DigiPerChannel = cms.untracked.bool(True), 
###                           DigiInFC = cms.untracked.bool(False),
###                          
###                           RecHitMonitor = cms.untracked.bool(True), 
###			   RecHitsPerChannel = cms.untracked.bool(True),
###
###                           ChannelQualityMonitor= cms.untracked.bool(True),
###                           nThreshold = cms.untracked.double(60),
###                           dThreshold = cms.untracked.double(1.0),
###                           OfflineMode = cms.untracked.bool(False),
###                           averageEnergyMethod = cms.untracked.bool(True),          
###
###                           PSMonitor= cms.untracked.bool(True),
###                           numberSigma = cms.untracked.double(1.5),
###                           thirdRegionThreshold =  cms.untracked.double(999999.0), # to avoid empty cells in reportSummary            
###                           EDMonitor= cms.untracked.bool(True),
###                           HIMonitor= cms.untracked.bool(True),
###                                       
###                           diagnosticPrescaleTime = cms.untracked.int32(-1),
###                           diagnosticPrescaleUpdate = cms.untracked.int32(-1),
###                           diagnosticPrescaleLS = cms.untracked.int32(-1),
###                             
###                           LEDMonitor = cms.untracked.bool(True),
###                           LEDPerChannel = cms.untracked.bool(True),
###                           FirstSignalBin = cms.untracked.int32(0),
###                           LastSignalBin = cms.untracked.int32(9),
###                           LED_ADC_Thresh = cms.untracked.double(-1000.0)      
###                           )

process.castorMonitor = cms.EDAnalyzer("CastorMonitorModule",
       ### GLOBAL VARIABLES
       debug = cms.untracked.int32(0), # make debug an int so that different
       # values can trigger different levels of messaging
       # 0 - no debug infomration
       # 1 - Program flow in/out
       # 2 - All major computations
       # Turn on/off timing diagnostic info
       showTiming = cms.untracked.bool(False),
       dump2database = cms.untracked.bool(False),
       pedestalsInFC = cms.untracked.bool(False),
       # Define Labels
       digiLabel = cms.InputTag("castorDigis"),
       rawLabel = cms.InputTag("rawDataCollector"), #rawDataCollector #source
       unpackerReportLabel = cms.InputTag("castorDigis"),
       CastorRecHitLabel = cms.InputTag("castorreco"),
       CastorTowerLabel = cms.InputTag("CastorTowerReco"),
       CastorBasicJetsLabel = cms.InputTag("ak7CastorJets"),
       CastorJetIDLabel = cms.InputTag("ak7CastorJetID"),
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
       thirdRegionThreshold = cms.untracked.double(999999.0),
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

print "Running with run type = ", process.runType.getRunTypeName()

if (process.runType.getRunTypeName() == process.runType.hi_run):
    process.castorDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.castorMonitor.rawLabel = cms.InputTag("rawDataRepacker")


### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
