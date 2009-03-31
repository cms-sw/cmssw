import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST')

process.source = cms.Source(
  "PoolSource",
  #fileNames = cms.untracked.vstring("/store/data/Commissioning08/Cosmics/RAW/v1/000/066/722/005AC873-4C9D-DD11-968E-000423D98F98.root")
  #fileNames = cms.untracked.vstring("/store/data/CRUZET3/Cosmics/RAW/v4/000/051/218/12E90673-E354-DD11-A5A5-001617C3B70E.root")
  #fileNames = cms.untracked.vstring("/store/data/CRUZET3/Cosmics/RAW/v1/000/050/900/A4ED1C68-1C4D-DD11-A472-000423D986A8.root")
)
process.load("DQM.SiStripMonitorHardware.test.source_cff")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

process.load('DQM.SiStripCommon.MessageLogger_cfi')

process.DQMStore = cms.Service("DQMStore")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT_30X::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

process.load('DQM.SiStripMonitorHardware.siStripFEDMonitor_cfi')
#process.load('DQM.SiStripMonitorHardware.siStripFEDMonitor_Tier0_cff')
#process.load('DQM.SiStripMonitorHardware.siStripFEDMonitor_P5_cff')
process.siStripFEDMonitor.PrintDebugMessages = True
process.siStripFEDMonitor.WriteDQMStore = True

process.Tracer = cms.Service('Tracer',indentation = cms.untracked.string('$$'))

process.p = cms.Path( process.siStripFEDMonitor )
