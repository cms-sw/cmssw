import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST')

process.source = cms.Source(
  "PoolSource",
  fileNames = cms.untracked.vstring("/store/data/CRUZET3/Cosmics/RAW/v4/000/051/218/12E90673-E354-DD11-A5A5-001617C3B70E.root")
  #fileNames = cms.untracked.vstring("/store/data/CRUZET3/Cosmics/RAW/v1/000/050/900/A4ED1C68-1C4D-DD11-A472-000423D986A8.root")
)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

process.load('DQM.SiStripCommon.MessageLogger_cfi')

process.DQMStore = cms.Service("DQMStore")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = "IDEAL_V9::All"
#process.GlobalTag.globaltag = "CRUZET4_V6H::All"
#process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
#process.prefer("GlobalTag")

process.load('DQM.SiStripMonitorHardware.siStripFEDCheck_cfi')
process.siStripFEDCheck.PrintDebugMessages = True
process.siStripFEDCheck.WriteDQMStore = True

process.Tracer = cms.Service('Tracer',indentation = cms.untracked.string('$$'))

process.p = cms.Path( process.siStripFEDCheck )
