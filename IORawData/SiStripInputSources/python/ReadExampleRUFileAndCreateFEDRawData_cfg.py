import FWCore.ParameterSet.Config as cms

process = cms.Process("ReadExampleRUFileAndCreateFEDRawData")
process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("IORawData.SiStripInputSources.ExampleRUFile_cff")

process.load("DQM.SiStripCommon.EventAnalyzer_cfi")

process.load("DQM.SiStripCommon.PoolOutput_cfi")

process.Timing = cms.Service("Timing")

process.p = cms.Path(process.EventAnalyzer)
process.e = cms.EndPath(process.PoolOutput)
process.maxEvents.input = 10

