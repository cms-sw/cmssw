# The following comments couldn't be translated into the new config version:

# suppresses html output from HCalClient  
import FWCore.ParameterSet.Config as cms

process = cms.Process("HCALDQM")
process.load("DQMServices.Core.DQM_cfg")

process.load("DQM.HcalMonitorModule.HcalMonitorModule_cfi")

process.load("DQM.HcalMonitorClient.HcalMonitorClient_cfi")

process.load("DQM.HcalMonitorModule.Hcal_FrontierConditions_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")

process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi")

process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi")

process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi")

#
# BEGIN DQM Online Environment #######################
#
# use include file for dqmEnv dqmSaver
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound', 
        'TooManyProducts', 
        'TooFewProducts')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("EventStreamHttpReader",
    sourceURL = cms.string('http://srv-c2c07-19.cms:11100/urn:xdaq-application:service=storagemanager'),
    consumerPriority = cms.untracked.string('normal'),
    max_event_size = cms.int32(7000000),
    consumerName = cms.untracked.string('HCAL DQM Consumer'),
    max_queue_depth = cms.int32(5),
    maxEventRequestRate = cms.untracked.double(10.0),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('PhysicsPath')
    ),
    headerRetryInterval = cms.untracked.int32(3)
)

process.p = cms.Path(process.hcalDigis*process.horeco*process.hfreco*process.hbhereco*process.hcalMonitor*process.hcalClient*process.dqmEnv*process.dqmSaver)
process.DQM.collectorHost = 'myhost'
process.DQM.collectorPort = 9092
process.hcalMonitor.DataFormatMonitor = True
process.hcalMonitor.DigiMonitor = True
process.hcalMonitor.HotCellMonitor = True
process.hcalMonitor.DeadCellMonitor = True
process.hcalMonitor.RecHitMonitor = True
process.hcalMonitor.TrigPrimMonitor = True
process.hcalMonitor.MTCCMonitor = False
process.hcalMonitor.PedestalMonitor = False
process.hcalMonitor.LEDMonitor = False
process.hcalMonitor.CaloTowerMonitor = False
process.hcalMonitor.HcalAnalysis = False
process.hcalClient.DataFormatClient = True
process.hcalClient.DigiClient = True
process.hcalClient.RecHitClient = True
process.hcalClient.HotCellClient = True
process.hcalClient.DeadCellClient = True
process.hcalClient.TrigPrimClient = True
process.hcalClient.CaloTowerClient = False
process.hcalClient.LEDClient = False
process.hcalClient.PedestalClient = False
process.hcalClient.baseHtmlDir = ''

process.GlobalTag.globaltag = 'CRUZET2_V2::All'
process.GlobalTag.connect = 'frontier://Frontier/CMS_COND_20X_GLOBALTAG' ##Frontier/CMS_COND_20X_GLOBALTAG" 

process.dqmSaver.convention = 'Online'
#replace dqmSaver.dirName          = "."
process.dqmSaver.producer = 'DQM'
process.dqmEnv.subSystemFolder = 'Hcal'

