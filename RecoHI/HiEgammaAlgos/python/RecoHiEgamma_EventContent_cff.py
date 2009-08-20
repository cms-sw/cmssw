# Full Ecal Event content 
RecoHiEgammaFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep recoSuperClusters_*_*_*',
    'keep recoCaloClusters_*_*_*',
    'keep recoClusterShapes_*_*_*',
    'keep recoCaloClustersToOnerecoClusterShapesAssociation_*_*_*',
    'keep *_ecalRecHit_*_*',
    )                                                                 
)

