import FWCore.ParameterSet.Config as cms

hltEgammaHGCALIDVarsL1Seeded = cms.EDProducer("EgammaHLTHGCalIDVarProducer",
    hgcalRecHits = cms.InputTag("hltParticleFlowRecHitHGCL1Seeded"),
    layerClusters = cms.InputTag("hltHgcalMergeLayerClustersL1Seeded"),
    rCylinder = cms.double(2.8),
    recoEcalCandidateProducer = cms.InputTag("hltEgammaCandidatesL1Seeded")
)
