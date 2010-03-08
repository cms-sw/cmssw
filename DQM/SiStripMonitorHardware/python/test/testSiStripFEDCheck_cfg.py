import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST')

process.source = cms.Source(
  "PoolSource",
  fileNames = cms.untracked.vstring(
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/129/200/A27810DC-F521-DF11-A7A2-001D09F291D7.root',
    )
  )
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

process.load('DQM.SiStripCommon.MessageLogger_cfi')

process.DQMStore = cms.Service("DQMStore")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "GR10_P_V1::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

process.load('DQM.SiStripMonitorHardware.siStripFEDCheck_cfi')
#process.siStripFEDCheck.PrintDebugMessages = False
process.siStripFEDCheck.WriteDQMStore = True
#process.siStripFEDCheck.HistogramUpdateFrequency = 1000
#process.siStripFEDCheck.DoPayloadChecks = True
#process.siStripFEDCheck.CheckChannelLengths = True
#process.siStripFEDCheck.CheckChannelPacketCodes = True
#process.siStripFEDCheck.CheckFELengths = True
#process.siStripFEDCheck.CheckChannelStatus = True

#process.Tracer = cms.Service('Tracer',indentation = cms.untracked.string('$$'))

process.p = cms.Path( process.siStripFEDCheck )
