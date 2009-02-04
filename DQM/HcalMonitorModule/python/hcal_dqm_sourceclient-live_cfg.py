# The following comments couldn't be translated into the new config version:

# suppresses html output from HCalClient  
import FWCore.ParameterSet.Config as cms

process = cms.Process("HCALDQM")
#----------------------------
# Event Source
#-----------------------------
#process.load("DQM.Integration.python.test.inputsource_cfi")
# This comes from DQM/Integration/python/test/inputsource_cfi.py
# How do we include a file from a subdirectory of 'python'?   Figure that out later -- Jeff T.
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

process.source = cms.Source("EventStreamHttpReader",
                            sourceURL = cms.string('http://cmsmon:50082/urn:xdaq-application:lid=29'),
                            consumerPriority = cms.untracked.string('normal'),
                            max_event_size = cms.int32(7000000),
                            consumerName = cms.untracked.string('DQM Source'),
                            max_queue_depth = cms.int32(5),
                            maxEventRequestRate = cms.untracked.double(15.0),
                            SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('*DQM')
                                                              ),
                            headerRetryInterval = cms.untracked.int32(3)
                            )


#----------------------------
# DQM Environment
#-----------------------------
process.load("DQMServices.Core.DQM_cfg")

#replace DQMStore.referenceFileName = "Hcal_reference.root"
process.load("DQMServices.Components.DQMEnvironment_cfi")

#----------------------------
# DQM Live Environment
#-----------------------------
#process.load("DQM.Integration.test.environment_cfi")
#process.load("DQM.Integration.python.test.environment_cfi")
# enviroment_cfi.py uses old format -- needs "process" in front of variables
process.DQM.collectorHost = 'srv-c2d05-19.cms'
process.DQM.collectorPort = 9090
process.dqmSaver.convention = 'Online'
process.dqmSaver.dirName = '/home/dqmprolocal/output'
process.dqmSaver.producer = 'Playback'
process.dqmSaver.saveByMinute = 200
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True




#-----------------------------
process.load("DQM.HcalMonitorModule.HcalMonitorModule_cfi")

process.load("DQM.HcalMonitorClient.HcalMonitorClient_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")

process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi")

process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi")

process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi")

process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_zdc_cfi")

process.prefer("GlobalTag")
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound', 
        'TooManyProducts', 
        'TooFewProducts')
)
process.p = cms.Path(process.hcalDigis*process.horeco*process.hfreco*process.hbhereco*process.zdcreco*process.hcalMonitor*process.hcalClient*process.dqmEnv*process.dqmSaver)
process.EventStreamHttpReader.consumerName = 'Hcal DQM Consumer'
process.dqmEnv.subSystemFolder = 'Hcal'
process.hcalMonitor.PedestalsPerChannel = False
process.hcalMonitor.PedestalsInFC = True
process.hcalMonitor.DataFormatMonitor = True
process.hcalMonitor.DigiMonitor = True
process.hcalMonitor.RecHitMonitor = True
process.hcalMonitor.TrigPrimMonitor = True
process.hcalMonitor.PedestalMonitor = True
process.hcalMonitor.DeadCellMonitor = True
process.hcalMonitor.HotCellMonitor = True
process.hcalMonitor.MTCCMonitor = False
process.hcalMonitor.LEDMonitor = False
process.hcalMonitor.CaloTowerMonitor = False
process.hcalMonitor.HcalAnalysis = False
process.hcalClient.plotPedRAW = True
process.hcalClient.baseHtmlDir = ''
process.hcalClient.SummaryClient = True
process.hcalClient.DataFormatClient = True
process.hcalClient.DigiClient = True
process.hcalClient.RecHitClient = True
process.hcalClient.DeadCellClient = True
process.hcalClient.TrigPrimClient = True
process.hcalClient.HotCellClient = True
process.hcalClient.CaloTowerClient = False
process.hcalClient.LEDClient = False
process.hcalClient.PedestalClient = False
process.GlobalTag.globaltag = 'CRUZET3_V6::All'
process.GlobalTag.connect = 'frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_20X_GLOBALTAG' ##(proxyurl=http:


