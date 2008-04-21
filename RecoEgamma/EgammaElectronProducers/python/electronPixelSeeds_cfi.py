import FWCore.ParameterSet.Config as cms

#
# module to produce pixel seeds for electrons from super clusters
# $Id: electronPixelSeeds.cfi,v 1.15 2008/04/08 16:38:53 uberthon Exp $
# Author:  Ursula Berthon, Claude Charlot
#
from RecoEgamma.EgammaElectronProducers.pixelSeedConfiguration_cfi import *
electronPixelSeeds = cms.EDProducer("ElectronPixelSeedProducer",
    endcapSuperClusters = cms.InputTag("correctedEndcapSuperClustersWithPreshower"),
    SeedConfiguration = cms.PSet(
        electronPixelSeedConfiguration
    ),
    SeedAlgo = cms.string(''),
    barrelSuperClusters = cms.InputTag("correctedHybridSuperClusters")
)


