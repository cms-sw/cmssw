import FWCore.ParameterSet.Config as cms

RecoHiEgammaFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep recoSuperClusters_*_*_*',
    'keep recoTracks_globalPrimTracks_*_*',
    'keep EcalRecHitsSorted_*_*_*',
    'keep recoVertexs_pixel3Vertices_*_*'
    'keep recoPhotons_photons_*_*',
    'keep floatedmValueMap_*_*_*',
    )
    )

RecoHiEgammaRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep recoSuperClusters_*_*_*',
    'keep recoTracks_globalPrimTracks_*_*',
    'keep EcalRecHitsSorted_*_*_*',
    'keep recoVertexs_pixel3Vertices_*_*'
    'keep *_hfreco_*_*',
    'keep *_horeco_*_*',
    'keep *_hbhereco_*_*',
    )
    )

RecoHiEgammaAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep recoPhotons_photons_*_*',
    'keep floatedmValueMap_*_*_*',
    'keep *_genParticles_*_*'
    )
    )
