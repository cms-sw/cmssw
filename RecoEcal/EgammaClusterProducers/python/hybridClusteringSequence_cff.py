import FWCore.ParameterSet.Config as cms

#------------------
#Hybrid clustering:
#------------------
# Producer for Hybrid BasicClusters and SuperClusters
from RecoEcal.EgammaClusterProducers.hybridSuperClusters_cfi import *
# Energy scale correction for Hybrid SuperClusters
from RecoEcal.EgammaClusterProducers.correctedHybridSuperClusters_cfi import *
# hybrid clustering sequence
#uncleanedHybridSuperClusters = RecoEcal.EgammaClusterProducers.hybridSuperClusters_cfi.hybridSuperClusters.clone()
uncleanedHybridSuperClusters = hybridSuperClusters.clone()
uncleanedHybridSuperClusters.RecHitSeverityToBeExcluded = cms.vint32(999)
uncleanedHybridSuperClusters.excludeFlagged = False

hybridClusteringSequence = cms.Sequence(hybridSuperClusters*correctedHybridSuperClusters * uncleanedHybridSuperClusters )

