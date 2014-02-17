import FWCore.ParameterSet.Config as cms

#
# $Id: hltIslandSuperClusters_cfi.py,v 1.2 2008/04/21 03:25:38 rpw Exp $
#
# Island SuperCluster producer
hltIslandSuperClusters = cms.EDProducer("SuperClusterProducer",
    barrelSuperclusterCollection = cms.string('islandBarrelSuperClusters'),
    endcapEtaSearchRoad = cms.double(0.14),
    barrelClusterCollection = cms.string('islandBarrelBasicClusters'),
    endcapClusterProducer = cms.string('hltIslandBasicClusters'),
    barrelPhiSearchRoad = cms.double(0.2),
    endcapPhiSearchRoad = cms.double(0.4),
    VerbosityLevel = cms.string('ERROR'),
    seedTransverseEnergyThreshold = cms.double(1.5),
    endcapSuperclusterCollection = cms.string('islandEndcapSuperClusters'),
    barrelEtaSearchRoad = cms.double(0.06),
    doBarrel = cms.bool(True),
    doEndcaps = cms.bool(True),
    endcapClusterCollection = cms.string('islandEndcapBasicClusters'),
    barrelClusterProducer = cms.string('hltIslandBasicClusters')
)


