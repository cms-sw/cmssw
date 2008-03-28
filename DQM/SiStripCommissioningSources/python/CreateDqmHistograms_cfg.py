import FWCore.ParameterSet.Config as cms

process = cms.Process("CreateDqmHistograms")
process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("DQM.SiStripCommon.DaqMonitorROOTBackEnd_cfi")

process.load("OnlineDB.SiStripConfigDb.SiStripConfigDb_cfi")

process.load("EventFilter.SiStripRawToDigi.FedChannelDigis_cfi")

process.load("DQM.SiStripCommissioningSources.CommissioningHistos_cfi")

process.FedCablingFromConfigDb = cms.ESSource("SiStripFedCablingBuilderFromDb",
    CablingSource = cms.untracked.string('UNDEFINED')
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:./input.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.p = cms.Path(process.FedChannelDigis*process.CommissioningHistos)
process.SiStripConfigDb.UsingDb = True
process.SiStripConfigDb.ConfDb = ''
process.SiStripConfigDb.Partition = ''
process.SiStripConfigDb.RunNumber = 0
process.FedChannelDigis.TriggerFedId = -1
process.CommissioningHistos.CommissioningTask = 'UNDEFINED'

