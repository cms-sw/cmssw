import FWCore.ParameterSet.Config as cms

#
# module to produce pixel seeds for electrons from super clusters
# $Id: electronPixelSeeds_cfi.py,v 1.6 2008/05/28 21:23:50 arizzi Exp $
# Author:  Ursula Berthon, Claude Charlot
#
from RecoEgamma.EgammaElectronProducers.pixelSeedConfiguration_cfi import *
electronPixelSeeds = cms.EDProducer("ElectronPixelSeedProducer",
    #  InputTag endcapSuperClusters = correctedEndcapSuperClustersWithPreshower
    endcapSuperClusters = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"),
    SeedConfiguration = cms.PSet(
        electronPixelSeedConfiguration,
        OrderedHitsFactoryPSet = cms.PSet(
            ComponentName = cms.string('StandardHitPairGenerator'),
            SeedingLayers = cms.string('MixedLayerPairs') ##"PixelLayerPairs"

        ),
        TTRHBuilder = cms.string('WithTrackAngle'),
        # eta-phi-region
        RegionPSet = cms.PSet(
            deltaPhiRegion = cms.double(0.7),
            originHalfLength = cms.double(15.0),
            useZInVertex = cms.bool(True),
            deltaEtaRegion = cms.double(0.3),
            ptMin = cms.double(1.5),
            originRadius = cms.double(0.2),
            VertexProducer = cms.InputTag("pixelVertices")
        )
    ),
    barrelSuperClusters = cms.InputTag("correctedHybridSuperClusters")
)


