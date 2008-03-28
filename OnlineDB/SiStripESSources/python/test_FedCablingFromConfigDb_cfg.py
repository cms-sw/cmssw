import FWCore.ParameterSet.Config as cms

process = cms.Process("test_FedCablingFromConfigDb")
process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("OnlineDB.SiStripESSources.FedCablingFromConfigDb_cff")

process.load("IORawData.SiStripInputSources.EmptySource_cff")

process.test = cms.EDFilter("test_FedCablingBuilder")

process.p = cms.Path(process.test)
process.FedCablingFromConfigDb.CablingSource = 'UNDEFINED'
process.maxEvents.input = 2

