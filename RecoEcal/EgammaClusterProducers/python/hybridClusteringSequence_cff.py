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
uncleanedHybridSuperClusters = cleanedHybridSuperClusters.clone()
uncleanedHybridSuperClusters.RecHitSeverityToBeExcluded = cms.vstring()
uncleanedHybridSuperClusters.excludeFlagged = False

from RecoEcal.EgammaClusterProducers.unifiedSCCollection_cfi import *

hybridClusteringSequence = cms.Sequence(
                cleanedHybridSuperClusters *
                uncleanedHybridSuperClusters *
                hybridSuperClusters *
                correctedHybridSuperClusters*
                uncleanedOnlyCorrectedHybridSuperClusters)

