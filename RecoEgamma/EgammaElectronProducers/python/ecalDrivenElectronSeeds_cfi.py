import FWCore.ParameterSet.Config as cms

#
# module to produce pixel seeds for electrons from super clusters
# $Id: ecalDrivenElectronSeeds_cfi.py,v 1.4 2011/02/15 23:35:42 vlimant Exp $
# Author:  Ursula Berthon, Claude Charlot
#
from RecoEgamma.EgammaElectronProducers.ecalDrivenElectronSeedsParameters_cff import *

ecalDrivenElectronSeeds = cms.EDProducer("ElectronSeedProducer",
    barrelSuperClusters = cms.InputTag("correctedHybridSuperClusters"),
    endcapSuperClusters = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"),
    SeedConfiguration = cms.PSet(
        ecalDrivenElectronSeedsParameters,
#        OrderedHitsFactoryPSet = cms.PSet(
#            ComponentName = cms.string('StandardHitPairGenerator'),
#            SeedingLayers = cms.string('MixedLayerPairs') 
#        ),
#        TTRHBuilder = cms.string('WithTrackAngle'),
#        # eta-phi region
#        RegionPSet = cms.PSet(
#            deltaPhiRegion = cms.double(0.7),
#            originHalfLength = cms.double(15.0),
#            useZInVertex = cms.bool(True),
#            deltaEtaRegion = cms.double(0.3),
#            ptMin = cms.double(1.5),
#            originRadius = cms.double(0.2),
#            VertexProducer = cms.InputTag("pixelVertices")
#        )
    )
)


