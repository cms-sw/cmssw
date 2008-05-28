import FWCore.ParameterSet.Config as cms

#
#  configuration for producer of converted photons
#  
#
softConversions = cms.EDProducer("SoftConversionProducer",
    clusterBarrelCollection = cms.string('islandBarrelBasicClusters'),
    outInTrackCollection = cms.string(''),
    softConversionCollection = cms.string('softConversionCollection'),
    clusterType = cms.string('BasicCluster'),
    conversionIOTrackProducer = cms.string('softConversionIOTracks'),
    outInTrackClusterAssociationCollection = cms.string('outInTrackClusterAssociationCollection'),
    clusterProducer = cms.string('islandBasicClusters'),
    clusterEndcapCollection = cms.string('islandEndcapBasicClusters'),
    inOutTrackClusterAssociationCollection = cms.string('inOutTrackClusterAssociationCollection'),
    inOutTrackCollection = cms.string(''),
    conversionOITrackProducer = cms.string('softConversionOITracks')
)



