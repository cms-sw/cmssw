import FWCore.ParameterSet.Config as cms

slimmedElectrons = cms.EDProducer("PATElectronSlimmer",
   src = cms.InputTag("selectedPatElectrons"),
   dropSuperCluster = cms.bool(False),
   dropBasicClusters = cms.bool(False),
   dropPFlowClusters = cms.bool(False),
   dropPreshowerClusters = cms.bool(False),
   dropRecHits = cms.bool(False),
)

