import FWCore.ParameterSet.Config as cms

isolationInputParameters = cms.PSet(
   barrelBasicCluster = cms.InputTag("islandBasicClusters","islandBarrelBasicClusters"),
   endcapBasicCluster = cms.InputTag("islandBasicClusters","islandEndcapBasicClusters"),
   horeco = cms.InputTag("horeco"),
   hfreco = cms.InputTag("hfreco"),
   hbhereco = cms.InputTag("hbhereco"),
   track = cms.InputTag("hiGeneralTracks"),
   photons = cms.InputTag("cleanPhotons")
   )
