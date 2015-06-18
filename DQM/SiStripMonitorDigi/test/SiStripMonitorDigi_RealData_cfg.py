import FWCore.ParameterSet.Config as cms

process = cms.Process("MonitorDigiRealData")

#--------------------------
# Event Source
#--------------------------
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#        '/store/data/Commissioning08/Cosmics/RAW/v1/000/067/647/0000721C-35A3-DD11-9132-001D09F291D7.root'
#       '/store/data/Commissioning08/Cosmics/RAW/v1/000/067/647/22CBBD11-07A3-DD11-9DFB-001D09F2447F.root'
#        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/668/ECBAB6B1-519C-DD11-BBB5-000423D94E70.root'
	"file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RAW__v1__165633__1CC420EE-B686-E011-A788-0030487CD6E8.root"
#"file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/Cosmics__RAW__v1__142560__026275A7-81A3-DF11-BDEE-001617C3B5D8.root"
)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

#-------------------------------------------------
# Message Logger
#-------------------------------------------------
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siStripDigis'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    destinations = cms.untracked.vstring('cout')
)

#-------------------------------------------------
# Geometry
#-------------------------------------------------
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-------------------------------------------------
# Calibration
#-------------------------------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR_R_44_V4::All'
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

#-----------------------
#  Reconstruction Modules
#-----------------------
process.load("EventFilter.SiStripRawToDigi.SiStripDigis_cfi")
process.siStripDigis.ProductLabel = 'source'

#--------------------------
# DQM Services
#--------------------------
process.DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(0)
)

process.TkDetMap = cms.Service("TkDetMap")
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

#--------------------------
# SiStrip MonitorDigi
#--------------------------
process.load("DQM.SiStripMonitorDigi.SiStripMonitorDigi_cfi")

process.SiStripMonitorDigi.CreateTrendMEs = True

process.SiStripMonitorDigi.TkHistoMap_On = True
process.SiStripMonitorDigi.TkHistoMapNApvShots_On = True
process.SiStripMonitorDigi.TkHistoMapNStripApvShots_On= True
process.SiStripMonitorDigi.TkHistoMapMedianChargeApvShots_On= True


process.SiStripMonitorDigi.TH1NApvShots.subdetswitchon = True
process.SiStripMonitorDigi.TH1NApvShots.globalswitchon = True

process.SiStripMonitorDigi.TH1ChargeMedianApvShots.subdetswitchon = True
process.SiStripMonitorDigi.TH1ChargeMedianApvShots.globalswitchon = True

process.SiStripMonitorDigi.TH1NStripsApvShots.subdetswitchon = True
process.SiStripMonitorDigi.TH1NStripsApvShots.globalswitchon = True

process.SiStripMonitorDigi.TH1ApvNumApvShots.subdetswitchon = True
process.SiStripMonitorDigi.TH1ApvNumApvShots.globalswitchon = True

process.SiStripMonitorDigi.TProfNShotsVsTime.subdetswitchon = True
process.SiStripMonitorDigi.TProfNShotsVsTime.globalswitchon = True

process.SiStripMonitorDigi.TProfTotalNumberOfDigis.subdetswitchon = True
process.SiStripMonitorDigi.TProfDigiApvCycle.subdetswitchon = True

#process.SiStripMonitorDigi.TH2DigiApvCycle.subdetswitchon = True
#process.SiStripMonitorDigi.TH2DigiApvCycle.yfactor = 0.005

process.SiStripMonitorDigi.SelectAllDetectors = True

process.SiStripMonitorDigi.TProfGlobalNShots.globalswitchon = True

process.outP = cms.OutputModule("AsciiOutputModule")
process.AdaptorConfig = cms.Service("AdaptorConfig")

#--------------------------
# Sequences 
#--------------------------

process.RecoForDQM = cms.Sequence(process.siStripDigis)

process.p = cms.Path(process.RecoForDQM*process.SiStripMonitorDigi)
process.ep = cms.EndPath(process.outP)


