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
# fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/data/Run2011A/MinimumBias/DQM/PromptReco-v2/000/163/765/E8ABE855-A375-E011-AF5F-0030487CD17C.root'
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/F6D43BCD-539C-DF11-8CBC-001D09F29524.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/E47E4419-4E9C-DF11-8F27-001D09F2424A.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/E409B15C-2C9C-DF11-9837-0030487C90C2.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/D8E72757-259C-DF11-A44B-000423D9997E.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/C8317183-4D9C-DF11-880B-001D09F2532F.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/C65AD16D-549C-DF11-B7C9-001D09F29114.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/BE085514-619C-DF11-8171-003048D2BDD8.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/B092D361-469C-DF11-8F73-001D09F2841C.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/AAF4065A-469C-DF11-86FE-003048F1183E.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/98967A7A-4F9C-DF11-95E1-001617E30D00.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/929EB6EF-559C-DF11-BA86-001D09F29169.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/7E1C9CDE-499C-DF11-8750-0030487CAF5E.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/66CFAC29-499C-DF11-9B3E-001D09F251FE.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/6612377E-4F9C-DF11-BAC7-0030487CD7C0.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/6043512C-2F9C-DF11-A695-001D09F2532F.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/5836E061-529C-DF11-817B-0030487CD6D2.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/5427A6EF-579C-DF11-9656-0030487CD6D8.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/484324DE-499C-DF11-9D9C-0030487CAEAC.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/46F3330A-479C-DF11-9D8F-001617C3B6FE.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/3AEB0426-289C-DF11-BA6F-0030487CAF0E.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/3A02246B-4D9C-DF11-A889-0030487A322E.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/32225742-449C-DF11-A7A1-0030487CD7C6.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/30F356CD-539C-DF11-B12F-001D09F25109.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/2223DE85-569C-DF11-A6E6-001617C3B6CE.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/16C74A14-2D9C-DF11-8705-0019B9F72BAA.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/1297BFDE-499C-DF11-AE99-001D09F24493.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/125C5760-4D9C-DF11-B62D-003048D2C1C4.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/10987364-529C-DF11-BBBF-001617C3B69C.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/0C226AA5-589C-DF11-905F-0030487C8E00.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/0A21F123-429C-DF11-AC2B-0030487C912E.root',
    '/store/data/Run2010A/MinimumBias/RECO/v4/000/142/035/00442033-909C-DF11-B291-001D09F24DA8.root'),
                            secondaryFileNames = cms.untracked.vstring(
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/FC2E509B-3E9C-DF11-A60C-001D09F25109.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/F0D0A2F8-2A9C-DF11-B17C-0030487C8CB6.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/E862A05E-339C-DF11-9FE0-001617C3B70E.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/E64AD8B5-409C-DF11-BFB3-003048D2BCA2.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/E6431A30-369C-DF11-BFC2-001D09F2424A.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/D6976FE8-2F9C-DF11-8A70-001D09F24934.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/D610C022-409C-DF11-92B1-0030487C778E.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/CCC76857-389C-DF11-8733-001D09F24498.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/CA766A1C-1A9C-DF11-AC55-0030487CD6D8.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/C08A4AAE-399C-DF11-A6AE-0019B9F709A4.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/BCDB9917-349C-DF11-9D97-001D09F24EAC.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/B88BB0AE-399C-DF11-993F-001D09F28EC1.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/A6175859-2C9C-DF11-B7B3-001617DC1F70.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/A2DA655A-2C9C-DF11-8D26-000423D9A2AE.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/9CB6B2E3-2F9C-DF11-8593-001D09F2AF96.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/963B894C-179C-DF11-BB99-0030487C7E18.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/92A8DDEA-3D9C-DF11-8E10-003048F024E0.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/8E016062-129C-DF11-BD6E-003048F024DE.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/720E3334-3B9C-DF11-922B-0030487CD716.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/6C89B9EA-3D9C-DF11-8780-003048F11C58.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/68DCC9A1-1D9C-DF11-A56C-003048F1110E.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/681CCC76-2E9C-DF11-BE06-001617E30D52.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/60AE6D6E-159C-DF11-9930-003048F1BF66.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/5AF74236-499C-DF11-95FD-0030487CAEAC.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/5A7F8C0D-329C-DF11-A8D8-001D09F254CE.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/581EED98-359C-DF11-B921-0030487CD178.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/3C02619C-379C-DF11-BD38-001D09F2438A.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/3AE8D082-1B9C-DF11-A42F-003048F110BE.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/22400431-3B9C-DF11-A52F-001D09F2AD4D.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/16672377-2E9C-DF11-91EB-001617E30D4A.root',
    '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/035/0C25C145-319C-DF11-BB60-001D09F24DA8.root')
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(510)
)


#================================
# DQM Environment
#================================
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
#process.DQMStore.referenceFileName = 'castor_reference.root'

process.load("Configuration.StandardSequences.Geometry_cff")

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
                           rawLabel             = cms.InputTag("source"),
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
                           thirdRegionThreshold =  cms.untracked.double(999999.0),        
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

