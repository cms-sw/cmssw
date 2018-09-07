import FWCore.ParameterSet.Config as cms

process = cms.Process("SiStripDigiAnalyzer")
process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("CalibTracker.Configuration.SiStrip_FakeConditions_cff")

process.load("IORawData.SiStripInputSources.EmptySource_cff")

process.load("EventFilter.SiStripRawToDigi.test.SiStripTrivialDigiSource_cfi")

process.load("EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi")

process.load("EventFilter.SiStripRawToDigi.SiStripDigis_cfi")

process.load("EventFilter.SiStripRawToDigi.test.SiStripDigiAnalyzer_cfi")

process.p = cms.Path(process.DigiSource*process.SiStripDigiToRaw*process.siStripDigis*process.DigiAnalyzer)
process.maxEvents.input = 2
process.SiStripDigiToRaw.InputDigis = cms.InputTag("DigiSource", "ZeroSuppressed")
process.siStripDigis.ProductLabel = 'SiStripDigiToRaw'

