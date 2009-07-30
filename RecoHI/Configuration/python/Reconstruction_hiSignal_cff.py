import FWCore.ParameterSet.Config as cms

from RecoHI.Configuration.Reconstruction_HI_cff import *

hiSignalGlobalPrimTracks = globalPrimTracks.clone()

hiSignalCorrectedIslandBarrelSuperClusters = correctedIslandBarrelSuperClusters.clone()
hiSignalCorrectedIslandEndcapSuperClusters = correctedIslandEndcapSuperClusters.clone()

hiSignalIterativeConePu5CaloJets = iterativeConePu5CaloJets.clone()

#Muon ?

#Write sequences
