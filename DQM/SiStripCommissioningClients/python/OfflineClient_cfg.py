import FWCore.ParameterSet.Config as cms

process = cms.Process("SiStripCommissioningOfflineClient")
process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("DQM.SiStripCommon.DaqMonitorROOTBackEnd_cfi")

process.load("IORawData.SiStripInputSources.EmptySource_cff")

process.client = cms.EDFilter("SiStripCommissioningOfflineClient",
    RunNumber = cms.untracked.uint32(0),
    UseClientFile = cms.untracked.bool(False),
    SummaryXmlFile = cms.untracked.FileInPath('DQM/SiStripCommissioningClients/data/summary.xml'),
    FilePath = cms.untracked.string('/tmp')
)

process.p = cms.Path(process.client)
process.maxEvents.input = 2

