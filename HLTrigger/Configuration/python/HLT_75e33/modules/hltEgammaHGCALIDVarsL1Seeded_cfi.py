import FWCore.ParameterSet.Config as cms

hltEgammaHGCALIDVarsL1Seeded = cms.EDProducer("EgammaHLTHGCalIDVarProducer",
    hgcalRecHits = cms.InputTag("particleFlowRecHitHGCL1Seeded"),
    layerClusters = cms.InputTag("hgcalMergeLayerClustersL1Seeded"),
    rCylinder = cms.double(2.8),
    recoEcalCandidateProducer = cms.InputTag("hltEgammaCandidatesL1Seeded")
)
# foo bar baz
# bQ5SWhuUl6l0J
# 3Zq4WLDZtuZIB
