import FWCore.ParameterSet.Config as cms

#------------------
#Hybrid clustering:
#------------------
# Producer for Hybrid BasicClusters and SuperClusters
from RecoEcal.EgammaClusterProducers.dynamicHybridSuperClusters_cfi import *
# Producer for energy corrections
from RecoEcal.EgammaClusterProducers.correctedDynamicHybridSuperClusters_cfi import *
# hybrid clustering sequence
dynamicHybridClusteringTask = cms.Task(dynamicHybridSuperClusters, correctedDynamicHybridSuperClusters)
dynamicHybridClusteringSequence = cms.Sequence(dynamicHybridClusteringTask)

