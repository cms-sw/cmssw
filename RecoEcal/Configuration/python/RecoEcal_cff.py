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

from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017

from RecoEcal.EgammaClusterProducers.islandClusteringSequence_cff import *

_ecalClustersHI = ecalClusters.copy()
_ecalClustersHI += islandClusteringSequence
for e in [pA_2016, peripheralPbPb, pp_on_AA_2018, pp_on_XeXe_2017, ppRef_2017]:
    e.toReplaceWith(ecalClusters, _ecalClustersHI)
