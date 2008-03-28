import FWCore.ParameterSet.Config as cms

process = cms.Process("SiStripCommissioningOfflineDbClient")
process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("DQM.SiStripCommon.DaqMonitorROOTBackEnd_cfi")

process.load("OnlineDB.SiStripConfigDb.SiStripConfigDb_cfi")

process.load("IORawData.SiStripInputSources.EmptySource_cff")

process.db_client = cms.EDFilter("SiStripCommissioningOfflineDbClient",
    SummaryXmlFile = cms.untracked.FileInPath('DQM/SiStripCommissioningClients/data/summary.xml'),
    UploadAnalyses = cms.untracked.bool(False),
    RunNumber = cms.untracked.uint32(0),
    UseClientFile = cms.untracked.bool(False),
    UploadHwConfig = cms.untracked.bool(False),
    FilePath = cms.untracked.string('/tmp')
)

process.p = cms.Path(process.db_client)
process.SiStripConfigDb.UsingDb = True
process.SiStripConfigDb.ConfDb = ''
process.SiStripConfigDb.Partition = ''
process.SiStripConfigDb.RunNumber = 0
process.maxEvents.input = 2

