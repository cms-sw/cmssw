import FWCore.ParameterSet.Config as cms

process = cms.Process("test_MessageLogger")
process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("IORawData.SiStripInputSources.EmptySource_cff")

process.test = cms.EDFilter("test_MessageLogger")

process.p = cms.Path(process.test)
process.maxEvents.input = 2

