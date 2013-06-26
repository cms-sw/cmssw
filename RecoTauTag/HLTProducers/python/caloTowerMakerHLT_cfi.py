import FWCore.ParameterSet.Config as cms

caloTowerMakerHLT = cms.EDProducer("CaloTowerCreatorForTauHLT",
    verbose = cms.untracked.int32(0),
    towers = cms.InputTag("towerMaker"),
    TauId = cms.int32(0),
    TauTrigger = cms.InputTag("l1extraParticles","Tau"),
    minimumE = cms.double(0.8),
    UseTowersInCone = cms.double(0.8),
    minimumEt = cms.double(0.5)
)


