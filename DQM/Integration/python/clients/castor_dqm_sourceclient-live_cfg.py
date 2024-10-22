from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import sys

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("CASTORDQM", Run3)

unitTest=False
if 'unitTest=True' in sys.argv:
    unitTest=True

#=================================
# Event Source
#================================+

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

#================================
# DQM Environment
#================================

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "Castor"
process.dqmSaver.tag = "Castor"
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = "Castor"
process.dqmSaverPB.runNumber = options.runNumber

process.load("FWCore.MessageLogger.MessageLogger_cfi")

#process.source = cms.Source("PoolSource",
#                            fileNames = cms.untracked.vstring(
#'file:/eos/user/p/popov/rundata/Castor2018/525D2460-6A90-E811-RAWrun320317.root'),
#                            )

#============================================
# Castor Conditions: from Global Conditions Tag 
#============================================
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
##
#from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
#process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run3_data', '')

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

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
process.castorMonitor = DQMEDAnalyzer("CastorMonitorModule",
   ### GLOBAL VARIABLES
   debug = cms.untracked.int32(0), #(=0 - no messages)
   # Turn on/off timing diagnostic
   showTiming          = cms.untracked.bool(False),
   # Define Labels
   l1tStage2uGtSource = cms.InputTag("gtStage2Digis"),
   tagTriggerResults   = cms.InputTag('TriggerResults','','HLT'),
   HltPaths  = cms.vstring("HLT_ZeroBias","HLT_Random"),
   digiLabel            = cms.InputTag("castorDigis"),
   rawLabel             = cms.InputTag("rawDataCollector"),
   unpackerReportLabel  = cms.InputTag("castorDigis"),
   CastorRecHitLabel    = cms.InputTag("castorreco"),
   CastorTowerLabel     = cms.InputTag("CastorTowerReco"),
   CastorBasicJetsLabel = cms.InputTag("ak7CastorJets"),
   CastorJetIDLabel     = cms.InputTag("ak7CastorJetID"),
   DataIntMonitor= cms.untracked.bool(True),
   TowerJetMonitor= cms.untracked.bool(True),
   DigiMonitor = cms.untracked.bool(True),
   RecHitMonitor = cms.untracked.bool(True),
#  LEDMonitor = cms.untracked.bool(True),
#  LEDPerChannel = cms.untracked.bool(True),
   FirstSignalBin = cms.untracked.int32(0),
   LastSignalBin = cms.untracked.int32(9)
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

process.p = cms.Path(process.castorDigis*process.castorreco*process.castorMonitor*process.dqmEnv*process.dqmSaver*process.dqmSaverPB)
#process.p = cms.Path(process.castorDigis*process.castorMonitor*process.dqmEnv*process.dqmSaver*process.dqmSaverPB)
#process.p = cms.Path(process.castorMonitor*process.dqmEnv*process.dqmSaver*process.dqmSaverPB)


process.castorDigis.InputLabel = "rawDataCollector"
process.castorMonitor.rawLabel = "rawDataCollector"
    
#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------

print("Running with run type = ", process.runType.getRunTypeName())

if (process.runType.getRunType() == process.runType.hi_run):
    process.castorDigis.InputLabel = "rawDataRepacker"
    process.castorMonitor.rawLabel = "rawDataRepacker"


### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
print("Final Source settings:", process.source)
process = customise(process)


