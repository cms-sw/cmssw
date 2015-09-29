import FWCore.ParameterSet.Config as cms

process = cms.Process("ReadExampleRUFileAndCreateDigis")
process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("IORawData.SiStripInputSources.ExampleRUFile_cfi")

process.load("IORawData.SiStripInputSources.FEDRawDataAnalyzer_cfi")

process.p = cms.Path(process.FedAnalyzer)
process.TBRUInputSource.maxEvents = 10
process.FedAnalyzer.pause_us = 0

