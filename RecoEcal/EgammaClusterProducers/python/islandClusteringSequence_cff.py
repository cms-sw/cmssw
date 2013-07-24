import FWCore.ParameterSet.Config as cms

#
# $Id: islandClusteringSequence_cff.py,v 1.2 2008/04/21 03:24:10 rpw Exp $
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

