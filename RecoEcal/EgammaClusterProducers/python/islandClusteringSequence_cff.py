import FWCore.ParameterSet.Config as cms

#
#
#------------------
#Island clustering:
#------------------
# Island BasicCluster producer
from RecoEcal.EgammaClusterProducers.islandBasicClusters_cfi import *
# Island SuperCluster producer
from RecoEcal.EgammaClusterProducers.islandSuperClusters_cfi import *
# Energy scale correction for Island SuperClusters
from RecoEcal.EgammaClusterProducers.correctedIslandBarrelSuperClusters_cfi import *
from RecoEcal.EgammaClusterProducers.correctedIslandEndcapSuperClusters_cfi import *
# create sequence for island clustering
islandClusteringSequence = cms.Sequence(islandBasicClusters*islandSuperClusters*correctedIslandBarrelSuperClusters*correctedIslandEndcapSuperClusters)

