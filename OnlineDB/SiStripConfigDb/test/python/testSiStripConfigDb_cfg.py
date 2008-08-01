import FWCore.ParameterSet.Config as cms

process = cms.Process("testSiStripConfigDb")

process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("OnlineDB.SiStripConfigDb.SiStripConfigDb_cfi")
process.SiStripConfigDb.UsingDb = True
process.SiStripConfigDb.ConfDb = ''

process.SiStripConfigDb.UsingDbCache = False
process.SiStripConfigDb.SharedMemory = ''

process.SiStripConfigDb.Partitions.PrimaryPartition.PartitionName = ''
process.SiStripConfigDb.Partitions.PrimaryPartition.RunNumber = 0

process.load("IORawData.SiStripInputSources.EmptySource_cff")
process.maxEvents.input = 2

process.test = cms.EDAnalyzer("testSiStripConfigDb",
    Upload               = cms.untracked.bool(False),
    Download             = cms.untracked.bool(False),
    FedConnections       = cms.untracked.bool(False),
    DeviceDescriptions   = cms.untracked.bool(False),
    FedDescriptions      = cms.untracked.bool(False),
    DcuDetIds            = cms.untracked.bool(False),
    AnalysisDescriptions = cms.untracked.bool(False),
)

process.p = cms.Path(process.test)

