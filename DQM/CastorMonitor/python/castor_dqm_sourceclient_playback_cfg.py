import FWCore.ParameterSet.Config as cms

process = cms.Process("CASTORDQM")
#=================================
# Event Source
#=================================
### process.load("DQM.Integration.test.inputsource_playback_cfi")
### process.EventStreamHttpReader.consumerName = 'Castor DQM Consumer'
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100000)
    )


#================================
# DQM Environment
#================================
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
#process.DQMStore.referenceFileName = 'castor_reference.root'

#================================
# DQM Playback Environment
#================================
process.load("DQM.Integration.test.desy_environment_playback_cfi")
process.dqmEnv.subSystemFolder    = "Castor"


#============================================
# Castor Conditions: from Global Conditions Tag 
#============================================
### process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
### process.GlobalTag.connect = "frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_21X_GLOBALTAG"
### process.GlobalTag.globaltag = 'CRAFT_V2H::All' # or any other appropriate
### process.prefer("GlobalTag")

process.load("FWCore.MessageLogger.MessageLogger_cfi")


#-----------------------------
# Castor DQM Source + SimpleReconstrctor
#-----------------------------
process.load("DQM.CastorMonitor.CastorMonitorModule_cfi")

process.load("EventFilter.CastorRawToDigi.CastorRawToDigi_cfi")

#process.load("RecoLocalCalo.CastorReco.CastorSimpleReconstructor_cfi") ##  Not yet in CMSSW !!!

process.castorMonitor.PedestalsPerChannel = True
process.castorMonitor.PedestalsInFC = False
#process.castorMonitor.checkNevents = 250

process.castorMonitor.PedestalMonitor = True
process.castorMonitor.RecHitMonitor = True
#process.castorMonitor.LEDMonitor = False

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

#process.p = cms.Path(process.castorDigis*process.castorreco*process.castorMonitor*process.dqmEnv*process.dqmSaver)
#process.p = cms.Path(process.castorDigis*process.castorMonitor*process.dqmEnv*process.dqmSaver)
process.p = cms.Path(process.castorMonitor*process.dqmEnv*process.dqmSaver)
