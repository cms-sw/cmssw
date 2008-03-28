import FWCore.ParameterSet.Config as cms

caloTowersPF = cms.EDFilter("CaloTowerCandidateCreator",
    src = cms.InputTag("towerMakerPF"),
    minimumEt = cms.double(-1.0),
    minimumE = cms.double(-1.0)
)


