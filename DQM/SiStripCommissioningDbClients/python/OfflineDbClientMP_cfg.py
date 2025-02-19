import FWCore.ParameterSet.Config as cms

process = cms.Process("SiStripCommissioningOfflineDbClientMP")

process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("DQM.SiStripCommon.DaqMonitorROOTBackEnd_cfi")

process.SiStripConfigDb = cms.Service("SiStripConfigDb",
    ConfDb  = cms.untracked.string(''),
    UsingDb = cms.untracked.bool(True),
    Partitions = cms.untracked.PSet(
        SecondaryPartition = cms.untracked.PSet(
            RunNumber     = cms.untracked.uint32(0),
            PartitionName = cms.untracked.string('')
        ),
        PrimaryPartition = cms.untracked.PSet(
            RunNumber     = cms.untracked.uint32(0),
            PartitionName = cms.untracked.string('')
        )
    )
)

process.load("IORawData.SiStripInputSources.EmptySource_cff")
process.maxEvents.input = 2

process.db_client = cms.EDAnalyzer(
    "SiStripCommissioningOfflineDbClient",
    FilePath         = cms.untracked.string('/tmp'),
    RunNumber        = cms.untracked.uint32(0),
    UseClientFile    = cms.untracked.bool(False),
    SummaryXmlFile   = cms.untracked.FileInPath('DQM/SiStripCommissioningClients/data/summary.xml'),
    UploadHwConfig   = cms.untracked.bool(False),
    UploadAnalyses   = cms.untracked.bool(False),
    DisableDevices   = cms.untracked.bool(False),
    DisableBadStrips = cms.untracked.bool(False),
    AddBadStrips 		= cms.untracked.bool(False),
    SaveClientFile = cms.untracked.bool(True)
    )

process.p = cms.Path(process.db_client)

