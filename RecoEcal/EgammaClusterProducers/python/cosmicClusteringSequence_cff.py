# The following comments couldn't be translated into the new config version:

#,
import FWCore.ParameterSet.Config as cms

#------------------
# clustering:
#------------------
#  BasicCluster producer
from RecoEcal.EgammaClusterProducers.cosmicBasicClusters_cfi import *
#  SuperCluster producer
#include "RecoEcal/EgammaClusterProducers/data/SuperClusters.cfi"
#  SuperCluster with Preshower producer
#include "RecoEcal/EgammaClusterProducers/data/SuperClustersWithPreshower.cfi"
# create sequence for  clustering
cosmicClusteringSequence = cms.Sequence(cosmicBasicClusters)

