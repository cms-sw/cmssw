import FWCore.ParameterSet.Config as cms

process = cms.Process("CreateDqmHistograms")

process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("DQM.SiStripCommon.DaqMonitorROOTBackEnd_cfi")

process.load("OnlineDB.SiStripConfigDb.SiStripConfigDb_cfi")
process.SiStripConfigDb.UsingDb = True
process.SiStripConfigDb.ConfDb  = ''
process.SiStripConfigDb.Partitions.PrimaryPartition.PartitionName = ''
process.SiStripConfigDb.Partitions.PrimaryPartition.RunNumber     = 0

process.FedCablingFromConfigDb = cms.ESSource("SiStripFedCablingBuilderFromDb",
    CablingSource = cms.untracked.string('UNDEFINED')
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:./input.root')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("EventFilter.SiStripRawToDigi.FedChannelDigis_cfi")
process.FedChannelDigis.TriggerFedId = -1

process.load("DQM.SiStripCommissioningSources.CommissioningHistos_cfi")
process.CommissioningHistos.CommissioningTask = 'UNDEFINED'

process.p = cms.Path(process.FedChannelDigis*process.CommissioningHistos)

