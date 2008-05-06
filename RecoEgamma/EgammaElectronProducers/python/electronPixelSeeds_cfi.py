import FWCore.ParameterSet.Config as cms

#
# module to produce pixel seeds for electrons from super clusters
# $Id: electronPixelSeeds_cfi.py,v 1.2 2008/04/21 03:24:48 rpw Exp $
# Author:  Ursula Berthon, Claude Charlot
#
from RecoEgamma.EgammaElectronProducers.pixelSeedConfiguration_cfi import *
electronPixelSeeds = cms.EDProducer("ElectronPixelSeedProducer",
    # endcapSuperClusters = cms.InputTag("correctedEndcapSuperClustersWithPreshower"),
    endcapSuperClusters = cms.InputTag("multi5x5SuperClustersWithPreshower"),
    SeedConfiguration = cms.PSet(
        electronPixelSeedConfiguration
    ),
    SeedAlgo = cms.string(''),
    barrelSuperClusters = cms.InputTag("correctedHybridSuperClusters"),
    RegionPSet = cms.PSet( ptMin = cms.double(1.5),
                           originRadius = cms.double(0.2),
                           originHalfLength = cms.double(15.0),
                           deltaEtaRegion = cms.double(0.3),
                           deltaPhiRegion = cms.double(0.7),
                           useZInVertex   = cms.bool(True),
                           VertexProducer = cms.InputTag('pixelVertices')
                         ),
    TTRHBuilder = cms.string('WithTrackAngle'),
    OrderedHitsFactoryPSet = cms.PSet( ComponentName = cms.string("StandardHitPairGenerator"),
                                       SeedingLayers = cms.string("MixedLayerPairs")
                                      )                                
                                                                      
)


