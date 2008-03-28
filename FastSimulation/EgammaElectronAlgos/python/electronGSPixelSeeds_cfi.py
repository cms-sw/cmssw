import FWCore.ParameterSet.Config as cms

#
# module to produce pixel seeds for electrons from super clusters
# $Id: electronGSPixelSeeds.cfi,v 1.6 2008/03/17 11:48:47 pjanot Exp $
# Author:  Ursula Berthon, Claude Charlot
#
from RecoEgamma.EgammaElectronProducers.pixelSeedConfiguration_cfi import *
electronGSPixelSeeds = cms.EDProducer("ElectronGSPixelSeedProducer",
    SeedConfiguration = cms.PSet(
        electronPixelSeedConfiguration
    ),
    # Inputs
    superClusterBarrel = cms.InputTag("correctedHybridSuperClusters"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    simTracks = cms.InputTag("famosSimHits"),
    SCEtCut = cms.double(5.0),
    # External seeding algorithm?
    SeedAlgo = cms.string(''),
    # A cut on the sim track transverse momentum (specific to fastSim)
    pTMin = cms.double(3.0),
    trackerHits = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits"),
    superClusterEndcap = cms.InputTag("correctedEndcapSuperClustersWithPreshower"),
    # for filtering
    hcalRecHits = cms.InputTag("caloRecHits"),
    maxHOverE = cms.double(0.2)
)


