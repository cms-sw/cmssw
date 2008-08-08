import FWCore.ParameterSet.Config as cms

process = cms.Process("HCALDQM")
process.load("DQMServices.Core.DQM_cfg")

# The number of events and file names will get overwritten when runoptions_cfi.py is loaded

###<<<<<<<<<< Don't remove this line -- it's used by the gui when updating/replacing file names!
process.maxEvents=cms.untracked.PSet(input = cms.untracked.int32(1000))
process.source = cms.Source("PoolSource",
	fileNames= cms.untracked.vstring(
		'/store/data/CRUZET3/Cosmics/RAW/v1/000/051/503/B8A8F592-7751-DD11-9823-000423D6C8EE.root',
		'/store/data/CRUZET3/Cosmics/RAW/v1/000/051/503/BC6FE216-7851-DD11-876B-000423D98804.root',
		'/store/data/CRUZET3/Cosmics/RAW/v1/000/051/503/485509C5-7751-DD11-8C1F-001617C3B706.root',
		'/store/data/CRUZET3/Cosmics/RAW/v1/000/051/503/188B380B-7A51-DD11-A3AA-001617DBCF90.root',
		'/store/data/CRUZET3/Cosmics/RAW/v1/000/051/503/84E447F7-7751-DD11-B54E-000423D6B444.root')
	)
###>>>>>>>>>>>  Don't remove this line!

process.load("DQM.HcalMonitorModule.HcalMonitorModule_cfi")

process.load("DQM.HcalMonitorClient.HcalMonitorClient_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")

process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi")

process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi")

process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi")

# Disabled for now; rewrite hcal_dqm... file directly (kind of dangerous?)
#process.load("DQM.HcalMonitorModule.dqmdbsGUI._runOptions_cfi")

#
# BEGIN DQM Online Environment ###########################
#
# use include file for dqmEnv dqmSaver
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.prefer("GlobalTag")
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound', 
        'TooManyProducts', 
        'TooFewProducts')
)

process.p = cms.Path(process.hcalDigis*process.horeco*process.hfreco*process.hbhereco*process.hcalMonitor*process.hcalClient*process.dqmEnv*process.dqmSaver)

process.DQM.collectorHost = 'myhost'
process.DQM.collectorPort = 9092

# hcalMonitor variables
process.hcalMonitor.DigiOccThresh = -999999999 ##Temporary measure while DigiOcc is reworked.
process.hcalMonitor.debug = False
process.hcalMonitor.DataFormatMonitor = True
process.hcalMonitor.DigiMonitor = True
process.hcalMonitor.PedestalMonitor = True
process.hcalMonitor.LEDMonitor = False
process.hcalMonitor.TrigPrimMonitor = True
process.hcalMonitor.HotCellMonitor = True
process.hcalMonitor.DeadCellMonitor = True
process.hcalMonitor.RecHitMonitor = True
process.hcalMonitor.MTCCMonitor = False
process.hcalMonitor.CaloTowerMonitor = False
process.hcalMonitor.HcalAnalysis = False
process.hcalMonitor.DigisPerChannel = False
process.hcalMonitor.PedestalsPerChannel = False
process.hcalMonitor.PedestalsInFC = True

#hcalClient variables -- set to same value as hcalMonitor
process.hcalClient.plotPedRAW = True
process.hcalClient.DoPerChanTests = False

process.hcalClient.SummaryClient = True
process.hcalClient.DataFormatClient = process.hcalMonitor.DataFormatMonitor
process.hcalClient.DigiClient = process.hcalMonitor.DigiMonitor
process.hcalClient.LEDClient = process.hcalMonitor.LEDMonitor
process.hcalClient.PedestalClient = process.hcalMonitor.PedestalMonitor
process.hcalClient.TrigPrimClient = process.hcalMonitor.TrigPrimMonitor
process.hcalClient.RecHitClient = process.hcalMonitor.RecHitMonitor
process.hcalClient.HotCellClient = process.hcalMonitor.HotCellMonitor
process.hcalClient.DeadCellClient = process.hcalMonitor.DeadCellMonitor
process.hcalClient.CaloTowerClient = process.hcalMonitor.CaloTowerMonitor

process.dqmSaver.convention = 'Online'
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'DQM'
process.dqmEnv.subSystemFolder = 'Hcal'


# Global Tags
# This works at FNAL, as of 8/8/08:
#This works at FNAL
process.GlobalTag.connect = 'frontier://Frontier/CMS_COND_21X_GLOBALTAG'
process.GlobalTag.globaltag = 'STARTUP_V4::All'
# For running at p5:
#process.GlobalTag.connect = "frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_20X_GLOBALTAG"
#process.GlobalTag.globaltag = 'CRUZET3_V6::All' # or any other appropriate
