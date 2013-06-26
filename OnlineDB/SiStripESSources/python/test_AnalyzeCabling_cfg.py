import FWCore.ParameterSet.Config as cms

process = cms.Process("testAnalyzeCabling")

process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("OnlineDB.SiStripESSources.FedCablingFromConfigDb_cff")
process.SiStripConfigDb.UsingDb = True
process.SiStripConfigDb.ConfDb  = ''
process.SiStripConfigDb.Partitions.PrimaryPartition.PartitionName = ''
process.SiStripConfigDb.Partitions.PrimaryPartition.RunNumber     = 0
process.FedCablingFromConfigDb.CablingSource = 'UNDEFINED'

process.load("IORawData.SiStripInputSources.EmptySource_cff")
process.maxEvents.input = 2

process.test = cms.EDAnalyzer("test_AnalyzeCabling")

process.p = cms.Path(process.test)

