import FWCore.ParameterSet.Config as cms

process = cms.Process("MonitorDigiRealData")

#--------------------------
# Event Source
#--------------------------
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#        '/store/data/Commissioning08/Cosmics/RAW/v1/000/067/647/0000721C-35A3-DD11-9132-001D09F291D7.root'
#       '/store/data/Commissioning08/Cosmics/RAW/v1/000/067/647/22CBBD11-07A3-DD11-9DFB-001D09F2447F.root'
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/668/ECBAB6B1-519C-DD11-BBB5-000423D94E70.root'
)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
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
process.load("Configuration.StandardSequences.Geometry_cff")

#-------------------------------------------------
# Calibration
#-------------------------------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "CRAFT_30X::All"
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
process.SiStripMonitorDigi.TkHistoMapDigi = True
process.SiStripMonitorDigi.OutputMEsInRootFile = True
process.SiStripMonitorDigi.TProfTotalNumberOfDigis.subdetswitchon = True
process.SiStripMonitorDigi.TProfDigiApvCycle.subdetswitchon = True
process.SiStripMonitorDigi.OutputFileName = 'SiStripMonitorDigi_RealData.root'
process.SiStripMonitorDigi.SelectAllDetectors = True

process.outP = cms.OutputModule("AsciiOutputModule")
process.AdaptorConfig = cms.Service("AdaptorConfig")

#--------------------------
# Sequences 
#--------------------------

process.RecoForDQM = cms.Sequence(process.siStripDigis)

process.p = cms.Path(process.RecoForDQM*process.SiStripMonitorDigi)
process.ep = cms.EndPath(process.outP)


