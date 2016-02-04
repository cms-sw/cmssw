import FWCore.ParameterSet.Config as cms

#
#  configuration for producer of converted photons
#  
#
softConversions = cms.EDProducer("SoftConversionProducer",
    clustersMaxDeltaPhi = cms.double(0.5),
    outInTrackCollection = cms.string(''),
    softConversionCollection = cms.string('softConversionCollection'),
    clusterType = cms.string('pfCluster'),
   # clusterType = cms.string('pfCluster'),
    #clusterBarrelCollection = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),
     clusterBarrelCollection = cms.InputTag("particleFlowClusterECAL"),                             
    conversionIOTrackProducer = cms.string('softConversionIOTracks'),
    outInTrackClusterAssociationCollection = cms.string('outInTrackClusterAssociationCollection'),
    clustersMaxDeltaEta = cms.double(0.2),
    clusterEndcapCollection = cms.InputTag("multi5x5BasicClusters","multi5x5EndcapBasicClusters"),
    inOutTrackClusterAssociationCollection = cms.string('inOutTrackClusterAssociationCollection'),
    trackMinHits = cms.double(3.0),
    inOutTrackCollection = cms.string(''),
    trackMaxChi2 = cms.double(100.0),
    conversionOITrackProducer = cms.string('softConversionOITracks')
)


