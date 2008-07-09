import FWCore.ParameterSet.Config as cms

HIRecoObjects = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_source_*_*', 
        'drop CrossingFrame_*_*_*', 
        'drop *_si*RecHits_*_*', 
        'drop *_*Digis_*_*', 
        'drop *_g4SimHits_*_*', 
        'drop *_si*Clusters_*_*', 
        'drop *_pixelTracks_*_*', 
        'drop *_*TrackCandidates_*_*', 
        'drop *_*TrackSeeds_*_*', 
        'drop *_hybridSuperClusters_*_*', 
        'drop *_islandSuperClusters_*_*', 
        'drop *_pixelTracksWithVertices_*_*')
)

