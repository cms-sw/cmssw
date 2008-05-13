import FWCore.ParameterSet.Config as cms

process = cms.Process("HCALDQM")
process.load("DQMServices.Core.DQM_cfg")

process.load("DQM.HcalMonitorModule.HcalMonitorModule_cfi")

process.load("DQM.HcalMonitorClient.HcalMonitorClient_cfi")

process.load("DQM.HcalMonitorModule.Hcal_FrontierConditions_GRuMM_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")

process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi")

process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi")

process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi")

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
#service = DaqMonitorROOTBackEnd{}
#
#service = MonitorDaemon{
#       #	 if true, will automatically start DQM thread in background
#       #  (default: false)
#       untracked bool AutoInstantiate = false
#
#       # if >=0, upon a connection problem, the source will automatically 
#       # attempt to reconnect with a time delay (secs) specified here 
#       # (default: 5)
#       untracked int32 reconnect_delay = 5
#       # maximum # of reconnection attempts upon connection problems (default: 10)
#       untracked int32 maxAttempts2Reconnect	= 0
#
#       # collector hostname
#       # (default: localhost)
#       untracked string DestinationAddress = "localhost"
#
#       # port for communicating with collector
#       # (default: 9090)
#       untracked int32 SendPort = 9090
#
#       # monitoring period in ms (i.e. how often monitoring elements 
#       # are shipped to the collector
#       # (default: 1000)
#       untracked int32 UpdateDelay = 10
#
#       # name of DQM source
#       # (default: DQMSource)
#       untracked string NameAsSource = "Hcal"
#}
#
#service = DQMShipMonitoring{
#	// event-period for shipping monitoring to collector (default: 25)
#	untracked uint32 period = 100
#}
#
# BEGIN DQM Online Environment #######################
#
# use include file for dqmEnv dqmSaver
process.load("DQMServices.Components.test.dqm_onlineEnv_cfi")

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound', 
        'TooManyProducts', 
        'TooFewProducts')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/121F13FE-E9A4-DC11-994D-001617DBD272.root', 
        '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/324F1DA3-E5A4-DC11-A732-000423D65A7E.root', 
        '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/34F36443-E9A4-DC11-8C8A-000423D944D4.root', 
        '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/360B4662-E8A4-DC11-B423-001617DBCEF6.root', 
        '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/421BAA91-E8A4-DC11-87A9-001617DBD36C.root', 
        '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/4CE78852-E6A4-DC11-B47A-000423D665B2.root', 
        '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/7637B20C-E8A4-DC11-950A-001617C3B652.root', 
        '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/7832F4A6-E6A4-DC11-AE02-00304856284E.root', 
        '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/943B3531-E7A4-DC11-B67C-001617DBD49A.root', 
        '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/A4C5514F-E5A4-DC11-9BDF-000423D98708.root', 
        '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/D0D20FF6-E7A4-DC11-BBDA-001617DBD2E2.root', 
        '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/D4486B09-E6A4-DC11-AA00-000E0C3F08B7.root', 
        '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/E0481BB5-E7A4-DC11-8BB2-003048561116.root', 
        '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/F016E2D2-E8A4-DC11-AD13-001617DBCFA6.root', 
        '/store/data/GlobalNov07/DTHcal/000/030/625/RAW/0000/F2A2C7F7-E6A4-DC11-BF27-000423D59C46.root')
)

process.p = cms.Path(process.hcalDigis*process.horeco*process.hfreco*process.hbhereco*process.hcalMonitor*process.hcalClient*process.dqmEnv*process.dqmSaver)
process.DQM.collectorHost = 'myhost'
process.DQM.collectorPort = 9092
process.PoolSource.fileNames = ['/store/data/GlobalMar08/A/000/000/000/RAW/0000/00565C56-ED07-DD11-909A-00304885AB96.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/00B0B16A-ED07-DD11-BC5A-001617E30CD4.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/0246C92B-EB07-DD11-A572-001617C3B710.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/02545322-E907-DD11-A3AA-003048560F10.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/042C445D-ED07-DD11-BA74-003048562936.root', 
    '/store/data/GlobalMar08/A/000/000/000/RAW/0000/0460DD46-EB07-DD11-9977-003048562986.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/04BA6152-ED07-DD11-B1CB-00304855D4CE.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/0608DD4A-ED07-DD11-A961-000423D98A44.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/064E5C4E-ED07-DD11-9C57-000423D98750.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/068B4B4A-ED07-DD11-BB46-000423D98DB4.root', 
    '/store/data/GlobalMar08/A/000/000/000/RAW/0000/06923D3A-EB07-DD11-B80E-000423D990CC.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/06C88232-EB07-DD11-A9D1-001617E30D38.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/08BBF948-EB07-DD11-BB91-00304885B0C2.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/0A1E3C0D-E907-DD11-B09D-000423D94990.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/0A8B432A-E907-DD11-BE1D-00304885AA4E.root', 
    '/store/data/GlobalMar08/A/000/000/000/RAW/0000/0AA3852D-EB07-DD11-98A6-00304855D4E8.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/0C03862F-EB07-DD11-BCD1-001617C3B614.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/0C11162A-E907-DD11-8CB3-003048562878.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/0C786D5D-ED07-DD11-A548-003048561206.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/0CC22915-E907-DD11-98A5-001617C3B6DC.root', 
    '/store/data/GlobalMar08/A/000/000/000/RAW/0000/0CEF0136-EB07-DD11-AA10-001617E30D40.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/0EF69608-E907-DD11-90D4-000423D991F0.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/104A250E-EB07-DD11-9B3B-001617E30D2C.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/1062BF38-EB07-DD11-BA2E-003048562880.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/10A26BFE-E807-DD11-94F1-000423DC1A0C.root', 
    '/store/data/GlobalMar08/A/000/000/000/RAW/0000/12147823-E907-DD11-A492-001617C3B614.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/1433DE26-EB07-DD11-9BAA-003048560F0E.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/147E345B-ED07-DD11-A7C0-003048562986.root', '/store/data/GlobalMar08/A/000/000/000/RAW/0000/14B28F0F-EB07-DD11-AA8A-001617C3B614.root']
process.dqmSaver.fileName = 'Hcal'
process.dqmEnv.subSystemFolder = 'Hcal'

