import FWCore.ParameterSet.Config as cms

process = cms.Process("test_FedCablingFromExampleDevices")
process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("OnlineDB.SiStripESSources.FedCablingFromExampleDevices_cff")

process.load("IORawData.SiStripInputSources.EmptySource_cff")

process.test = cms.EDFilter("test_FedCablingBuilder")

process.p = cms.Path(process.test)
process.maxEvents.input = 2

