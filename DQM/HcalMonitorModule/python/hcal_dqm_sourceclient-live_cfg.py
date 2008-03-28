# The following comments couldn't be translated into the new config version:

#        ecalRecHit, towerMaker,  

import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQM.HcalMonitorModule.HcalMonitorModule_cfi")

process.load("DQM.HcalMonitorClient.HcalMonitorClient_live_cfi")

process.load("DQM.HcalMonitorModule.Hcal_FrontierConditions_GREN_ORCON_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")

process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi")

process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi")

process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi")

#
# BEGIN DQM Online Environment #######################
#
# use include file for dqmEnv dqmSaver
process.load("DQMServices.Components.test.dqm_onlineEnv_cfi")

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound', 'TooManyProducts', 'TooFewProducts')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("EventStreamHttpReader",
    sourceURL = cms.string('http://cmsdisk1.cms:11100/urn:xdaq-application:service=storagemanager'),
    consumerPriority = cms.untracked.string('normal'),
    max_event_size = cms.int32(7000000),
    consumerName = cms.untracked.string('HCAL DQM Consumer'),
    max_queue_depth = cms.int32(5),
    maxEventRequestRate = cms.untracked.double(10.0),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('*')
    ),
    headerRetryInterval = cms.untracked.int32(3)
)

process.DaqMonitorROOTBackEnd = cms.Service("DaqMonitorROOTBackEnd",
    verbose = cms.untracked.int32(0)
)

process.MonitorDaemon = cms.Service("MonitorDaemon",
    #	 if true, will automatically start DQM thread in background
    #  (default: false)
    AutoInstantiate = cms.untracked.bool(True),
    # collector hostname
    # (default: localhost)
    DestinationAddress = cms.untracked.string('srv-c2d05-19'),
    # maximum     # of reconnection attempts upon connection problems (default: 10)
    maxAttempts2Reconnect = cms.untracked.int32(99999),
    # port for communicating with collector
    # (default: 9090)
    SendPort = cms.untracked.int32(9090),
    # name of DQM source
    # (default: DQMSource)
    NameAsSource = cms.untracked.string('Hcal'),
    # monitoring period in ms (i.e. how often monitoring elements 
    # are shipped to the collector
    # (default: 1000)
    UpdateDelay = cms.untracked.int32(10),
    # if >=0, upon a connection problem, the source will automatically 
    # attempt to reconnect with a time delay (secs) specified here 
    # (default: 5)
    reconnect_delay = cms.untracked.int32(5)
)

process.DQMShipMonitoring = cms.Service("DQMShipMonitoring",
    # event-period for shipping monitoring to collector (default: 25)
    period = cms.untracked.uint32(25)
)

process.p = cms.Path(process.hcalDigis*process.horeco*process.hfreco*process.hbhereco*process.hcalMonitor*process.hcalClient*process.dqmEnv*process.dqmSaver)
process.dqmSaver.fileName = 'Hcal'
process.dqmSaver.dirName = '/data1/dropbox'
process.dqmEnv.subSystemFolder = 'Hcal'

