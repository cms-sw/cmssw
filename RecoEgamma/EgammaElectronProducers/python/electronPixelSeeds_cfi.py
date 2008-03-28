import FWCore.ParameterSet.Config as cms

#
# module to produce pixel seeds for electrons from super clusters
# $Id: electronPixelSeeds.cfi,v 1.12 2008/02/07 13:15:35 uberthon Exp $
# Author:  Ursula Berthon, Claude Charlot
#
from RecoEgamma.EgammaElectronProducers.pixelSeedConfiguration_cfi import *
electronPixelSeeds = cms.EDProducer("ElectronPixelSeedProducer",
    SeedConfiguration = cms.PSet(
        electronPixelSeedConfiguration
    ),
    superClusterBarrelProducer = cms.string('correctedHybridSuperClusters'),
    superClusterBarrelLabel = cms.string(''),
    SeedAlgo = cms.string(''),
    superClusterEndcapLabel = cms.string(''),
    superClusterEndcapProducer = cms.string('correctedEndcapSuperClustersWithPreshower')
)


