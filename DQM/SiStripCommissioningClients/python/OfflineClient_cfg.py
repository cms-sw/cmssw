import FWCore.ParameterSet.Config as cms

process = cms.Process("SiStripCommissioningOfflineClient")

process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("DQM.SiStripCommon.DaqMonitorROOTBackEnd_cfi")

process.load("IORawData.SiStripInputSources.EmptySource_cff")
process.maxEvents.input = 2

process.client = cms.EDFilter("SiStripCommissioningOfflineClient",
    FilePath       = cms.untracked.string('/tmp')
    RunNumber      = cms.untracked.uint32(0),
    UseClientFile  = cms.untracked.bool(False),
    SummaryXmlFile = cms.untracked.FileInPath('DQM/SiStripCommissioningClients/data/summary.xml'),
    SaveClientFile = cms.untracked.bool(True),
)

process.p = cms.Path(process.client)

