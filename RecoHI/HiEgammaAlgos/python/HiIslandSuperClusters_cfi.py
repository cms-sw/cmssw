import FWCore.ParameterSet.Config as cms

#
#
# Island SuperCluster producer
islandSuperClusters = cms.EDProducer("HiSuperClusterProducer",
    barrelSuperclusterCollection = cms.string('islandBarrelSuperClusters'),
    endcapEtaSearchRoad = cms.double(0.14),
    barrelClusterCollection = cms.string('islandBarrelBasicClusters'),
    endcapClusterProducer = cms.string('islandBasicClusters'),
    barrelPhiSearchRoad = cms.double(0.8),
    endcapPhiSearchRoad = cms.double(0.6),
    VerbosityLevel = cms.string('ERROR'),
    seedTransverseEnergyThreshold = cms.double(1.0),
    barrelBCEnergyThreshold = cms.double(0),
    endcapBCEnergyThreshold = cms.double(0),
    endcapSuperclusterCollection = cms.string('islandEndcapSuperClusters'),
    barrelEtaSearchRoad = cms.double(0.07),
    doBarrel = cms.bool(True),
    doEndcaps = cms.bool(True),
    endcapClusterCollection = cms.string('islandEndcapBasicClusters'),
    barrelClusterProducer = cms.string('islandBasicClusters')
)


