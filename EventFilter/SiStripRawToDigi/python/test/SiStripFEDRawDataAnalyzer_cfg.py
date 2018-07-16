import FWCore.ParameterSet.Config as cms

process = cms.Process("SiStripFEDRawDataAnalyzer")
process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("CalibTracker.Configuration.SiStrip_FakeConditions_cff")

process.load("IORawData.SiStripInputSources.EmptySource_cff")

process.load("EventFilter.SiStripRawToDigi.test.SiStripTrivialClusterSource_cfi")

process.load("EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi")

process.load("EventFilter.SiStripRawToDigi.test.SiStripFEDRawDataAnalyzer_cfi")

process.p = cms.Path(process.ClusterSource*process.SiStripDigiToRaw*process.FEDRawDataAnalyzer)
process.maxEvents.input = 2
process.SiStripDigiToRaw.InputDigis = cms.InputTag('ClusterSource', "ZeroSuppressed")
process.FEDRawDataAnalyzer.InputLabel = 'SiStripDigiToRaw'

