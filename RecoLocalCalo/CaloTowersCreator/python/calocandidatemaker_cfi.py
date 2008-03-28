import FWCore.ParameterSet.Config as cms

caloTowerCandidates = cms.EDFilter("CaloTowerCandidateCreator",
    src = cms.InputTag("towermaker"),
    # Minimum transverse energy for the tower to be converted to a candidate (GeV)
    minimumEt = cms.double(-1.0),
    # Minimum energy for the tower to be converted to a candidate (GeV)
    minimumE = cms.double(-1.0)
)


