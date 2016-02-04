import FWCore.ParameterSet.Config as cms

#------------------
#FixedMatrix clustering:
#------------------
# FixedMatrix BasicCluster producer
from RecoEcal.EgammaClusterProducers.fixedMatrixBasicClusters_cfi import *
# FixedMatrix SuperCluster producer
from RecoEcal.EgammaClusterProducers.fixedMatrixSuperClusters_cfi import *
# FixedMatrix SuperCluster with Preshower producer
from RecoEcal.EgammaClusterProducers.fixedMatrixSuperClustersWithPreshower_cfi import *
# create sequence for fixedMatrix clustering
fixedMatrixClusteringSequence = cms.Sequence(fixedMatrixBasicClusters*fixedMatrixSuperClusters*fixedMatrixSuperClustersWithPreshower)

