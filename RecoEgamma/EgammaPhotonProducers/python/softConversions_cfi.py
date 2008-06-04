import FWCore.ParameterSet.Config as cms

#
#  configuration for producer of converted photons
#  
#
softConversions = cms.EDProducer("SoftConversionProducer",
    clustersMaxDeltaPhi = cms.double(0.5),
    clusterBarrelCollection = cms.string('islandBarrelBasicClusters'),
    outInTrackCollection = cms.string(''),
    softConversionCollection = cms.string('softConversionCollection'),
    clusterType = cms.string('BasicCluster'),
    conversionIOTrackProducer = cms.string('softConversionIOTracks'),
    outInTrackClusterAssociationCollection = cms.string('outInTrackClusterAssociationCollection'),
    clustersMaxDeltaEta = cms.double(0.2),
    clusterProducer = cms.string('islandBasicClusters'),
    clusterEndcapCollection = cms.string('islandEndcapBasicClusters'),
    inOutTrackClusterAssociationCollection = cms.string('inOutTrackClusterAssociationCollection'),
    trackMinHits = cms.double(3.0),
    inOutTrackCollection = cms.string(''),
    trackMaxChi2 = cms.double(100.0),
    conversionOITrackProducer = cms.string('softConversionOITracks')
)


