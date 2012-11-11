import FWCore.ParameterSet.Config as cms

# Sequence for clustering in ecal barrel & endcap + preshower
# hybrid clustering sequence
from RecoEcal.EgammaClusterProducers.hybridClusteringSequence_cff import *
# multi5x5 clusters
from RecoEcal.EgammaClusterProducers.multi5x5ClusteringSequence_cff import *
# preshower sequence for multi5x5 clusters
from RecoEcal.EgammaClusterProducers.multi5x5PreshowerClusteringSequence_cff import *
#reduced recHit collection producer
from RecoEcal.EgammaClusterProducers.reducedRecHitsSequence_cff import *

#create the EcalNextToDeadChannel record on the fly
from RecoEcal.EgammaCoreTools.EcalNextToDeadChannelESProducer_cff import *
# create path with all clustering algos
# NB: preshower MUST be run after multi5x5 clustering in the endcap

#particle flow super clustering sequence
from RecoEcal.EgammaClusterProducers.particleFlowSuperClusteringSequence_cff import *

ecalClustersNoPFBox = cms.Sequence(hybridClusteringSequence*multi5x5ClusteringSequence*multi5x5PreshowerClusteringSequence)
ecalClusters = cms.Sequence(ecalClustersNoPFBox*particleFlowSuperClusteringSequence)
