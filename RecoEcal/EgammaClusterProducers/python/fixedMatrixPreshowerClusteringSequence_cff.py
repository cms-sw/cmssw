import FWCore.ParameterSet.Config as cms

#
#
#------------------
#Preshower clustering:
#------------------
# producer for endcap SuperClusters including preshower energy
from RecoEcal.EgammaClusterProducers.correctedFixedMatrixSuperClustersWithPreshower_cfi import *
# producer for preshower cluster shapes
from RecoEcal.EgammaClusterProducers.fixedMatrixPreshowerClusterShape_cfi import *
# create sequence for preshower clustering
fixedMatrixPreshowerClusteringTask = cms.Task(correctedFixedMatrixSuperClustersWithPreshower,fixedMatrixPreshowerClusterShape)
fixedMatrixPreshowerClusteringSequence = cms.Sequence(fixedMatrixPreshowerClusteringTask)

# foo bar baz
# GBW2ZOOW1DdzU
# Sqb9sWJnbU85S
