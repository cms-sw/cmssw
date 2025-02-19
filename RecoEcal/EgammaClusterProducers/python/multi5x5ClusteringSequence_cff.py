import FWCore.ParameterSet.Config as cms

#------------------
#Multi5x5 clustering:
#------------------
# Multi5x5 BasicCluster producer
from RecoEcal.EgammaClusterProducers.multi5x5BasicClusters_cfi import *
# Multi5x5 SuperCluster producer
from RecoEcal.EgammaClusterProducers.multi5x5SuperClusters_cfi import *
# Multi5x5 SuperCluster with Preshower producer
from RecoEcal.EgammaClusterProducers.multi5x5SuperClustersWithPreshower_cfi import *
# create sequence for multi5x5 clustering
multi5x5ClusteringSequence = cms.Sequence(multi5x5BasicClustersCleaned*
                                          multi5x5SuperClustersCleaned*
                                          multi5x5BasicClustersUncleaned*
                                          multi5x5SuperClustersUncleaned*
                                          #now unify clean and unclean  
                                          multi5x5SuperClusters*
                                          multi5x5SuperClustersWithPreshower
                                          )

