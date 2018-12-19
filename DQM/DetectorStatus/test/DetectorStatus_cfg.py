import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("DQM.Integration.config.fileinputsource_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("DQM.DetectorStatus.detectorstatus_cfi")
process.DQMStore = cms.Service("DQMStore")

process.load("DQMServices.FileIO.DQMFileSaverOnline_cfi")

process.p = cms.Path(process.detectorstatus + process.dqmSaver)
