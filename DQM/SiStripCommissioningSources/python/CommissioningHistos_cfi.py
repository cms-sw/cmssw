import FWCore.ParameterSet.Config as cms

CommissioningHistos = cms.EDFilter("SiStripCommissioningSource",
    SummaryInputModuleLabel = cms.string('FedChannelDigis'),
    RootFileName = cms.untracked.string('SiStripCommissioningSource'),
    CommissioningTask = cms.untracked.string('UNDEFINED'),
    InputModuleLabel = cms.string('FedChannelDigis'),
    HistoUpdateFreq = cms.untracked.int32(10)
)


