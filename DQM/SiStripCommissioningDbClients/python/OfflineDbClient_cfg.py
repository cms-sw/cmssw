import FWCore.ParameterSet.Config as cms

process = cms.Process("SiStripCommissioningOfflineDbClient")

process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("DQM.SiStripCommon.DaqMonitorROOTBackEnd_cfi")

process.load("OnlineDB.SiStripConfigDb.SiStripConfigDb_cfi")
process.SiStripConfigDb.UsingDb = True
process.SiStripConfigDb.ConfDb  = ''
process.SiStripConfigDb.Partitions.PrimaryPartition.PartitionName = ''
process.SiStripConfigDb.Partitions.PrimaryPartition.RunNumber     = 0

process.load("IORawData.SiStripInputSources.EmptySource_cff")
process.maxEvents.input = 2

process.db_client = cms.EDAnalyzer("SiStripCommissioningOfflineDbClient",
    FilePath       = cms.untracked.string('/tmp'),
    RunNumber      = cms.untracked.uint32(0),
    UseClientFile  = cms.untracked.bool(False),
    SummaryXmlFile = cms.untracked.FileInPath('DQM/SiStripCommissioningClients/data/summary.xml'),
    UploadHwConfig = cms.untracked.bool(False),
    UploadAnalyses = cms.untracked.bool(False),
    DisableDevices = cms.untracked.bool(False),
    DisableStrips  = cms.untracked.bool(False),
    SaveClientFile = cms.untracked.bool(True)
)

process.p = cms.Path(process.db_client)

