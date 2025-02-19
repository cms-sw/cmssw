import FWCore.ParameterSet.Config as cms

CommissioningHistos = cms.EDAnalyzer("SiStripCommissioningSource",
  InputModuleLabel         = cms.string('FedChannelDigis'),
  SummaryInputModuleLabel  = cms.string('FedChannelDigis'),
  HistoUpdateFreq          = cms.untracked.int32(500),
  RootFileName             = cms.untracked.string('SiStripCommissioningSource'),
  CommissioningTask        = cms.untracked.string('UNDEFINED'),
  View                     = cms.untracked.string('Default'),
  PedsFullNoiseParameters  = cms.PSet(
    NrEvToSkipAtStart    = cms.int32(1000),
    NrEvForPeds          = cms.int32(4000),
    NrPosBinsNoiseHist   = cms.int32(50),
    FillNoiseProfile     = cms.bool(False),
    UseAverageCommonMode = cms.bool(True),  # expert use only - default : true
    UseFloatPedestals    = cms.bool(True)   # expert use only - default : true
  )
)

