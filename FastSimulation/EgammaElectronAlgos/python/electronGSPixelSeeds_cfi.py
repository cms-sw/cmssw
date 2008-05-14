import FWCore.ParameterSet.Config as cms

#
# module to produce pixel seeds for electrons from super clusters
# $Id: electronGSPixelSeeds.cfi,v 1.9 2008/04/29 18:06:54 pjanot Exp $
# Author:  Ursula Berthon, Claude Charlot
#
from RecoEgamma.EgammaElectronProducers.pixelSeedConfiguration_cfi import *
electronGSPixelSeeds = cms.EDProducer("ElectronGSPixelSeedProducer",
    endcapSuperClusters = cms.InputTag("correctedEndcapSuperClustersWithPreshower"),
    SeedConfiguration = cms.PSet(
        electronPixelSeedConfiguration
    ),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    # Inputs
    barrelSuperClusters = cms.InputTag("correctedHybridSuperClusters"),
    simTracks = cms.InputTag("famosSimHits"),
    SCEtCut = cms.double(5.0),
    # A cut on the sim track transverse momentum (specific to fastSim)
    pTMin = cms.double(3.0),
    trackerHits = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits"),
    # for filtering
    hcalRecHits = cms.InputTag("caloRecHits"),
    maxHOverE = cms.double(0.2)
)


