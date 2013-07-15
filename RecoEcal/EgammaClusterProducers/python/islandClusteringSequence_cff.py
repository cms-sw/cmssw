import FWCore.ParameterSet.Config as cms

#
# $Id: islandClusteringSequence.cff,v 1.7 2007/03/13 17:21:44 futyand Exp $
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

