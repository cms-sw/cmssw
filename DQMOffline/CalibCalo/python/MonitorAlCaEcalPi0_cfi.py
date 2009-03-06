import FWCore.ParameterSet.Config as cms

EcalPi0Mon = cms.EDAnalyzer("DQMSourcePi0",
    AlCaStreamEBTag = cms.untracked.InputTag("hltAlCaPi0RegRecHits","pi0EcalRecHitsEB"),
    SaveToFile = cms.untracked.bool(False),
    FolderName = cms.untracked.string('ALCAStreamEcalCalPi0'),
    AlCaStreamEETag = cms.untracked.InputTag("hltAlCaPi0RegRecHits","pi0EcalRecHitsEE"),
    isMonEB = cms.untracked.bool(True),
    prescaleFactor = cms.untracked.int32(1),
    FileName = cms.untracked.string('MonitorAlCaEcalPi0.root'),
    isMonEE = cms.untracked.bool(False)
)



