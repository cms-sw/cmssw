import FWCore.ParameterSet.Config as cms

#
# module to produce pixel seeds for electrons from super clusters
# $Id: electronGSPixelSeeds.cfi,v 1.11 2008/05/29 13:40:55 beaudett Exp $
# Author:  Ursula Berthon, Claude Charlot
#
from RecoEgamma.EgammaElectronProducers.pixelSeedConfiguration_cfi import *
electronGSPixelSeeds = cms.EDProducer("ElectronGSPixelSeedProducer",
    endcapSuperClusters = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"),
    SeedConfiguration = cms.PSet(
        electronPixelSeedConfiguration
    ),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    # Inputs
    barrelSuperClusters = cms.InputTag("correctedHybridSuperClusters"),
    simTracks = cms.InputTag("famosSimHits"),
    # A cut on the sim track transverse momentum (specific to fastSim)
    pTMin = cms.double(3.0),
    trackerHits = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
)


