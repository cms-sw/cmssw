# Full Ecal Event content 
RecoHIEgammaFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep *_islandBasicClusters_*_*',
    'keep *_multi5x5BasicClusters_*_*',
    'keep *_ecalRecHit_*_*',
    'keep *_ecalPreshowerRecHit_*_*',
    'keep *_simEcalTriggerPrimitiveDigis_*_*',
    'keep *_ecalDigis_*_*',
    'keep *_ecalGlobalUncalibRecHit_*_*',
    'keep *_hfreco_*_*',
    'keep *_horeco_*_*',
    'keep *_hbhereco_*_*',
    'keep *_globalPrimTracks_*_*',
    'keep *_photons_*_*',
    'keep *_photonCore_*_*',
    'keep *_genParticles_*_*',
    'keep *_hiGenParticles_*_*'
    'keep *_pixel3Vertices_*_*'

    )                                                                 
)

