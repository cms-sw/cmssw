import FWCore.ParameterSet.Config as cms

EcalPi0Mon = cms.EDFilter("DQMSourcePi0",
    seleS4S9GammaOne = cms.double(0.85),
    prescaleFactor = cms.untracked.int32(1),
    seleMinvMaxPi0 = cms.double(0.22),
    gammaCandPhiSize = cms.int32(21),
    selePtGammaOne = cms.double(0.9),
    ParameterX0 = cms.double(0.89),
    FolderName = cms.untracked.string('HLT/AlCaEcalPi0'),
    selePtPi0 = cms.double(2.5),
    clusSeedThr = cms.double(0.5),
    AlCaStreamEBTag = cms.untracked.InputTag("hltAlCaPi0RegRecHits","pi0EcalRecHitsEB"),
    SaveToFile = cms.untracked.bool(False),
    clusPhiSize = cms.int32(3),
    selePi0BeltDR = cms.double(0.2),
    clusEtaSize = cms.int32(3),
    isMonEE = cms.untracked.bool(False),
    selePi0Iso = cms.double(0.5),
    ParameterW0 = cms.double(4.2),
    seleNRHMax = cms.int32(1000),
    selePi0BeltDeta = cms.double(0.05),
    isMonEB = cms.untracked.bool(True),
    ParameterLogWeighted = cms.bool(True),
    seleXtalMinEnergy = cms.double(0.0),
    seleS4S9GammaTwo = cms.double(0.85),
    seleMinvMinPi0 = cms.double(0.06),
    gammaCandEtaSize = cms.int32(9),
    FileName = cms.untracked.string('MonitorAlCaEcalPi0.root'),
    AlCaStreamEETag = cms.untracked.InputTag("hltAlCaPi0RegRecHits","pi0EcalRecHitsEE"),
    selePtGammaTwo = cms.double(0.9),
    ParameterT0_barl = cms.double(5.7)
)



