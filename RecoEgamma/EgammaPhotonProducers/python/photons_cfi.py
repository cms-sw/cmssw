import FWCore.ParameterSet.Config as cms

#
# producer for photons
# $Id: photons.cfi,v 1.18 2008/04/15 14:37:47 nancy Exp $
#
photons = cms.EDProducer("PhotonProducer",
    scHybridBarrelProducer = cms.string('correctedHybridSuperClusters'),
    endcapHitProducer = cms.string('ecalRecHit'),
    minR9 = cms.double(0.93),
    usePrimaryVertex = cms.bool(True),
    scIslandEndcapProducer = cms.string('correctedEndcapSuperClustersWithPreshower'),
    primaryVertexProducer = cms.string('offlinePrimaryVerticesFromCTFTracks'),
    conversionCollection = cms.string(''),
    endcapClusterShapeMapCollection = cms.string('islandEndcapShapeAssoc'),
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    posCalc_t0_endcPresh = cms.double(3.6),
    posCalc_logweight = cms.bool(True),
    scIslandEndcapCollection = cms.string(''),
    barrelClusterShapeMapProducer = cms.string('hybridSuperClusters'),
    posCalc_w0 = cms.double(4.2),
    photonCollection = cms.string(''),
    pixelSeedProducer = cms.string('electronPixelSeeds'),
    conversionProducer = cms.string('conversions'),
    hbheInstance = cms.string(''),
    scHybridBarrelCollection = cms.string(''),
    hbheModule = cms.string('hbhereco'),
    posCalc_t0_endc = cms.double(6.3),
    barrelClusterShapeMapCollection = cms.string('hybridShapeAssoc'),
    minSCEt = cms.double(5.0),
    maxHOverE = cms.double(1.0),
    hOverEConeSize = cms.double(0.1),
    posCalc_x0 = cms.double(0.89),
    endcapHitCollection = cms.string('EcalRecHitsEE'),
    endcapClusterShapeMapProducer = cms.string('islandBasicClusters'),
    barrelHitProducer = cms.string('ecalRecHit'),
    posCalc_t0_barl = cms.double(7.7)
)


