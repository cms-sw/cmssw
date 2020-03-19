import FWCore.ParameterSet.Config as cms

from RecoHI.Configuration.Reconstruction_HI_cff import *

# Make sure the intermediate digi+reco products of the input even are dropped.
# Add this in your PoolSource.inputCommands

dropAllExceptFinalProducts = cms.PSet()

# Clone the final product producers, whose originals will be kept from the previous mixed event.

#Tracks
hiSignalGlobalPrimTracks = hiGlobalPrimTracks.clone()
hiSignalSelectedTracks = hiSelectedTracks.clone()
hiSelectedTracks.src = cms.InputTag("hiSignalGlobalPrimTracks")
heavyIonTrackingTask = cms.Task(hiPixelVerticesTask
                                ,hiPrimSeedsTask
                                ,hiPrimTrackCandidates
                                ,hiSignalGlobalPrimTracks
                                ,hiSelectedTracks
                                )
heavyIonTracking = cms.Sequence(heavyIonTrackingTask)

#Ecal
hiSignalCorrectedIslandBarrelSuperClusters = correctedIslandBarrelSuperClusters.clone()
hiSignalCorrectedIslandEndcapSuperClusters = correctedIslandEndcapSuperClusters.clone()

islandClusteringTask = cms.Task(islandBasicClusters
                                ,islandSuperClusters
                                ,hiSignalCorrectedIslandBarrelSuperClusters
                                ,hiSignalCorrectedIslandEndcapSuperClusters
                                )
islandClusteringSequence = cms.Sequence(islandClusteringTask)

#Jets
hiSignalIterativeConePu5CaloJets = iterativeConePu5CaloJets.clone()
runjetsTask = cms.Task(caloTowersRecTask,caloTowers,hiSignalIterativeConePu5CaloJets)
runjets = cms.Sequence(runjetsTask)

#Muons
hiSignalGlobalMuons = globalMuons.clone()
hiSignalGlobalMuons.TrackerCollectionLabel = 'hiSignalGlobalPrimTracks'
muontrackingTask = cms.Task(standAloneMuonSeedsTask,standAloneMuons,hiSignalGlobalMuons)
muontracking = cms.Sequence(muontrackingTask)

#Use same sequences as Reconstruction_HI_cff

