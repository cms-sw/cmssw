import FWCore.ParameterSet.Config as cms

process = cms.Process("test_PedestalsFromExampleXmlFile")
process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("OnlineDB.SiStripESSources.PedestalsFromExampleXmlFile_cff")

process.load("IORawData.SiStripInputSources.EmptySource_cff")

process.test = cms.EDAnalyzer("test_PedestalsBuilder")

process.p = cms.Path(process.test)
process.maxEvents.input = 2

