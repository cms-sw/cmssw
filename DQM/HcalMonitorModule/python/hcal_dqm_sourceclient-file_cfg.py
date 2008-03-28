# The following comments couldn't be translated into the new config version:

#        ecalRecHit, towerMaker,  

# To run on .dat streamer files (you need to change the path to the datafile):
# source = NewEventStreamFileReader {
# 	untracked vstring fileNames = {
# 	    "file:/data1/lookarea/GlobalAug07.00017220.0001.A.storageManager.0.0000.dat",
# 	    "file:/data1/lookarea/GlobalAug07.00017220.0002.A.storageManager.0.0000.dat",
# 	    "file:/data1/lookarea/GlobalAug07.00017220.0003.A.storageManager.0.0000.dat"
# 	}
# }
#To run on HCAL local runs:  (you need to have acces to /bigspool or modify the path to the datafile)	
# source = HcalTBSource {
#        untracked vstring fileNames = {'file:/bigspool/usc/USC_035349.root'}
#        untracked vstring streams = { 
# 		//HBHE a, b, c:
# 		'HCAL_DCC700','HCAL_DCC701','HCAL_DCC702','HCAL_DCC703','HCAL_DCC704','HCAL_DCC705',
# 		'HCAL_DCC706','HCAL_DCC707','HCAL_DCC708','HCAL_DCC709','HCAL_DCC710','HCAL_DCC711',
# 		'HCAL_DCC712','HCAL_DCC713','HCAL_DCC714','HCAL_DCC715','HCAL_DCC716','HCAL_DCC717',
# 		//HF:
# 		'HCAL_DCC718','HCAL_DCC719','HCAL_DCC720','HCAL_DCC721','HCAL_DCC722','HCAL_DCC723',
# 		//HO:
# 		'HCAL_DCC724','HCAL_DCC725','HCAL_DCC726','HCAL_DCC727','HCAL_DCC728','HCAL_DCC729',
# 		'HCAL_DCC730','HCAL_DCC731',
# 		'HCAL_Trigger','HCAL_SlowData'
# 	}
# }	

import FWCore.ParameterSet.Config as cms

process = cms.Process("HCALDQM")
process.load("DQM.HcalMonitorModule.HcalMonitorModule_cfi")

process.load("DQM.HcalMonitorClient.HcalMonitorClient_cfi")

process.load("DQM.HcalMonitorModule.Hcal_FrontierConditions_GREN_cff")

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
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/121F13FE-E9A4-DC11-994D-001617DBD272.root', '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/324F1DA3-E5A4-DC11-A732-000423D65A7E.root', '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/34F36443-E9A4-DC11-8C8A-000423D944D4.root', '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/360B4662-E8A4-DC11-B423-001617DBCEF6.root', '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/421BAA91-E8A4-DC11-87A9-001617DBD36C.root', '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/4CE78852-E6A4-DC11-B47A-000423D665B2.root', '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/7637B20C-E8A4-DC11-950A-001617C3B652.root', '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/7832F4A6-E6A4-DC11-AE02-00304856284E.root', '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/943B3531-E7A4-DC11-B67C-001617DBD49A.root', '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/A4C5514F-E5A4-DC11-9BDF-000423D98708.root', '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/D0D20FF6-E7A4-DC11-BBDA-001617DBD2E2.root', '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/D4486B09-E6A4-DC11-AA00-000E0C3F08B7.root', '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/E0481BB5-E7A4-DC11-8BB2-003048561116.root', '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/F016E2D2-E8A4-DC11-AD13-001617DBCFA6.root', '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/F2A2C7F7-E6A4-DC11-BF27-000423D59C46.root')
)

process.DaqMonitorROOTBackEnd = cms.Service("DaqMonitorROOTBackEnd")

process.MonitorDaemon = cms.Service("MonitorDaemon",
    #	 if true, will automatically start DQM thread in background
    #  (default: false)
    AutoInstantiate = cms.untracked.bool(False),
    # collector hostname
    # (default: localhost)
    DestinationAddress = cms.untracked.string('localhost'),
    # maximum     # of reconnection attempts upon connection problems (default: 10)
    maxAttempts2Reconnect = cms.untracked.int32(0),
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
    period = cms.untracked.uint32(100)
)

process.p = cms.Path(process.hcalDigis*process.horeco*process.hfreco*process.hbhereco*process.hcalMonitor*process.hcalClient*process.dqmEnv*process.dqmSaver)
process.dqmSaver.fileName = 'Hcal'
process.dqmEnv.subSystemFolder = 'Hcal'

