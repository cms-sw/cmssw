import FWCore.ParameterSet.Config as cms

#
# module to produce pixel seeds for electrons from super clusters
# $Id: electronPixelSeeds_cfi.py,v 1.2 2008/04/21 03:24:48 rpw Exp $
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


