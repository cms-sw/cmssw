# Full Ecal Event content 
RecoHIEgammaFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep recoSuperClusters_*_*_*',
    'keep recoCaloClusters_*_*_*',
    'keep recoClusterShapes_*_*_*',
    'keep recoCaloClustersToOnerecoClusterShapesAssociation_*_*_*',
    'keep *_ecalRecHit_*_*',
    'keep *_ecalPreshowerRecHit_*_*',
    'keep *_hfreco_*_*',
    'keep *_horeco_*_*',
    'keep *_hbhereco_*_*',
    'keep *_globalPrimTracks_*_*',
    'keep *_photons_*_*',
    'keep *_photonCore_*_*',
    'keep *_genParticles_*_*',
    'keep *_hiGenParticles_*_*',
    'keep edmHepMCProduct_*_*_*'
    

    )                                                                 
)

