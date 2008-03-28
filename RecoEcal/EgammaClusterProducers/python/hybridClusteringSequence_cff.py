import FWCore.ParameterSet.Config as cms

#------------------
#Hybrid clustering:
#------------------
# Producer for Hybrid BasicClusters and SuperClusters
from RecoEcal.EgammaClusterProducers.hybridSuperClusters_cfi import *
# Energy scale correction for Hybrid SuperClusters
from RecoEcal.EgammaClusterProducers.correctedHybridSuperClusters_cfi import *
# hybrid clustering sequence
hybridClusteringSequence = cms.Sequence(hybridSuperClusters*correctedHybridSuperClusters)

