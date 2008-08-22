import FWCore.ParameterSet.Config as cms

process = cms.Process("HCALDQM")
process.load("DQMServices.Core.DQM_cfg")

# The number of events and file names will get overwritten when runoptions_cfi.py is loaded

###<<<<<<<<<< Don't remove this line -- it's used by the gui when updating/replacing file names!
process.maxEvents=cms.untracked.PSet(input = cms.untracked.int32(200))
process.source = cms.Source("PoolSource",
	fileNames= cms.untracked.vstring(
		'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/764/8CEF3AF9-256E-DD11-A619-001617E30E2C.root',
		'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/764/5A5AA7D6-256E-DD11-B3B4-001617E30F48.root',
		'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/764/C6884D22-266E-DD11-89AC-000423D985E4.root',
		'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/764/5A5AA7D6-256E-DD11-B3B4-001617E30F48.root',
		'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/764/0A9EEDFA-266E-DD11-A83C-001617C3B64C.root',
		'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/764/5803EA38-2A6E-DD11-9F44-001617C3B6E8.root',
		'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/764/5A5AA7D6-256E-DD11-B3B4-001617E30F48.root',
		'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/764/92C93CF9-256E-DD11-AF88-001617E30D4A.root',
		'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/764/5C529842-246E-DD11-9C5F-0019DB29C614.root',
		'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/764/92C93CF9-256E-DD11-AF88-001617E30D4A.root',
		'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/764/C6884D22-266E-DD11-89AC-000423D985E4.root',
		'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/764/8CEF3AF9-256E-DD11-A619-001617E30E2C.root',
		'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/764/5C529842-246E-DD11-9C5F-0019DB29C614.root',
		'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/764/0A9EEDFA-266E-DD11-A83C-001617C3B64C.root',
		'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/764/8CEF3AF9-256E-DD11-A619-001617E30E2C.root')
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

process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_zdc_cfi")

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

process.p = cms.Path(process.hcalDigis*process.horeco*process.hfreco*process.hbhereco*process.zdcreco*process.hcalMonitor*process.hcalClient*process.dqmEnv*process.dqmSaver)

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
process.GlobalTag.globaltag = 'CRUZET4_V2::All'

# For running at p5:
#process.GlobalTag.connect = "frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_20X_GLOBALTAG"
#process.GlobalTag.globaltag = 'CRUZET3_V6::All' # or any other appropriate
