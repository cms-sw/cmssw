import FWCore.ParameterSet.Config as cms

#
#  configuration for producer of converted photons
#  
#
conversions = cms.EDProducer("ConvertedPhotonProducer",
    scHybridBarrelProducer = cms.string('correctedHybridSuperClusters'),
    scHybridBarrelCollection = cms.string(''),
    convertedPhotonCollection = cms.string(''),
    scIslandEndcapProducer = cms.string('multi5x5SuperClustersWithPreshower'),
    inOutTrackSCAssociation = cms.string('inOutTrackSCAssociationCollection'),
    outInTrackCollection = cms.string(''),
    barrelClusterShapeMapCollection = cms.string('hybridShapeAssoc'),
    bcEndcapCollection = cms.string('islandEndcapBasicClusters'),
    conversionIOTrackProducer = cms.string('ckfInOutTracksFromConversions'),
    bcBarrelCollection = cms.string('islandBarrelBasicClusters'),
    inOutTrackCollection = cms.string(''),
    bcProducer = cms.string('islandBasicClusters'),
    outInTrackSCAssociation = cms.string('outInTrackSCAssociationCollection'),
    scIslandEndcapCollection = cms.string(''),
    barrelClusterShapeMapProducer = cms.string('hybridSuperClusters'),
    conversionOITrackProducer = cms.string('ckfOutInTracksFromConversions')
)


