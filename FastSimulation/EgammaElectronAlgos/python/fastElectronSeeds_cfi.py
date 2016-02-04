import FWCore.ParameterSet.Config as cms

#
# module to produce pixel seeds for electrons from super clusters
# $Id: fastElectronSeeds_cfi.py,v 1.1 2009/02/04 11:05:16 chamont Exp $
# Author:  Ursula Berthon, Claude Charlot
#
from RecoEgamma.EgammaElectronProducers.ecalDrivenElectronSeedsParameters_cff import *

fastElectronSeeds = cms.EDProducer("FastElectronSeedProducer",
    endcapSuperClusters = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"),
    SeedConfiguration = cms.PSet(
        ecalDrivenElectronSeedsParameters
    ),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    # Inputs
    barrelSuperClusters = cms.InputTag("correctedHybridSuperClusters"),
    simTracks = cms.InputTag("famosSimHits"),
    # A cut on the sim track transverse momentum (specific to fastSim)
    pTMin = cms.double(3.0),
    trackerHits = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
)


