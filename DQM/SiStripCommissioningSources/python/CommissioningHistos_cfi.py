import FWCore.ParameterSet.Config as cms

CommissioningHistos = cms.EDAnalyzer("SiStripCommissioningSource",
  InputModuleLabel         = cms.string('FedChannelDigis'),
  SummaryInputModuleLabel  = cms.string('FedChannelDigis'),
  HistoUpdateFreq          = cms.untracked.int32(10),
  RootFileName             = cms.untracked.string('SiStripCommissioningSource'),
  CommissioningTask        = cms.untracked.string('UNDEFINED'),
  View                     = cms.untracked.string('Default'),
  PedsFullNoiseParameters  = cms.PSet(
    NrEvToSkipAtStart  = cms.untracked.int32(0),
    NrEvUntilStable    = cms.untracked.int32(20),
    NrPosBinsNoiseHist = cms.untracked.int32(25)
  )
)

