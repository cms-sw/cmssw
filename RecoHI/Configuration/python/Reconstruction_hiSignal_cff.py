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
heavyIonTracking = cms.Sequence(hiPixelVertices
                                * hiPrimSeeds
                                * hiPrimTrackCandidates
                                * hiSignalGlobalPrimTracks
                                * hiSelectedTracks
                                )

#Ecal
hiSignalCorrectedIslandBarrelSuperClusters = correctedIslandBarrelSuperClusters.clone()
hiSignalCorrectedIslandEndcapSuperClusters = correctedIslandEndcapSuperClusters.clone()

islandClusteringSequence = cms.Sequence(islandBasicClusters
                                        *islandSuperClusters
                                        *hiSignalCorrectedIslandBarrelSuperClusters
                                        *hiSignalCorrectedIslandEndcapSuperClusters
                                        )

#Jets
hiSignalIterativeConePu5CaloJets = iterativeConePu5CaloJets.clone()
runjets = cms.Sequence(caloTowersRec*caloTowers*hiSignalIterativeConePu5CaloJets)

#Muons
hiSignalGlobalMuons = globalMuons.clone()
hiSignalGlobalMuons.TrackerCollectionLabel = 'hiSignalGlobalPrimTracks'
muontracking = cms.Sequence(standAloneMuonSeeds*standAloneMuons*hiSignalGlobalMuons)


#Use same sequences as Reconstruction_HI_cff

