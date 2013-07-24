import FWCore.ParameterSet.Config as cms

#
# $Id: HiIslandClusteringSequence_cff.py,v 1.1 2010/10/26 10:29:41 yjlee Exp $
#
#------------------
#Island clustering:
#------------------
# Island BasicCluster producer
from RecoEcal.EgammaClusterProducers.islandBasicClusters_cfi import *
# Island SuperCluster producer
from RecoHI.HiEgammaAlgos.HiIslandSuperClusters_cfi import *
# Energy scale correction for Island SuperClusters
from RecoHI.HiEgammaAlgos.HiCorrectedIslandBarrelSuperClusters_cfi import *
from RecoHI.HiEgammaAlgos.HiCorrectedIslandEndcapSuperClusters_cfi import *
# create sequence for island clustering
islandClusteringSequence = cms.Sequence(islandBasicClusters*islandSuperClusters*correctedIslandBarrelSuperClusters*correctedIslandEndcapSuperClusters)

