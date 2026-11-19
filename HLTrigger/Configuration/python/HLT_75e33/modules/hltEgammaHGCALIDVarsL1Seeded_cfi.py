import FWCore.ParameterSet.Config as cms

hltEgammaHGCALIDVarsL1Seeded = cms.EDProducer("EgammaHLTHGCalIDVarProducer",
    ticlGeom = cms.ESInputTag('', 'hlt'),
    hgcalRecHits = cms.InputTag("hltParticleFlowRecHitHGCL1Seeded"),
    layerClusters = cms.InputTag("hltMergeLayerClustersL1Seeded"),
    rCylinder = cms.double(2.8),
    recoEcalCandidateProducer = cms.InputTag("hltEgammaCandidatesL1Seeded")
)
