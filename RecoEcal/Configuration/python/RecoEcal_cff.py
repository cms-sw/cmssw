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

ecalClustersNoPFBoxTask = cms.Task(hybridClusteringTask,
                              multi5x5ClusteringTask,
                              multi5x5PreshowerClusteringTask)
ecalClustersNoPFBox = cms.Sequence(ecalClustersNoPFBoxTask)
ecalClustersTask = cms.Task(ecalClustersNoPFBoxTask, particleFlowSuperClusteringTask)
ecalClusters = cms.Sequence(ecalClustersTask)

from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017

from RecoEcal.EgammaClusterProducers.islandClusteringSequence_cff import *

_ecalClustersHITask = ecalClustersTask.copy()
_ecalClustersHITask.add(islandClusteringTask)
for e in [pA_2016, peripheralPbPb, pp_on_AA, pp_on_XeXe_2017, ppRef_2017]:
    e.toReplaceWith(ecalClustersTask, _ecalClustersHITask)
