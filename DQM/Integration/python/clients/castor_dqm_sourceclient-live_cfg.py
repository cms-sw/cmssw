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
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
##
#from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
#process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run2_data', '')

#-----------------------------
# Castor DQM Source + SimpleReconstrctor
#-----------------------------
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

from EventFilter.CastorRawToDigi.CastorRawToDigi_cff import *
process.castorDigis = castorDigis.clone()

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

if (process.runType.getRunType() == process.runType.hi_run):
    process.castorDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.castorMonitor.rawLabel = cms.InputTag("rawDataRepacker")


### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
