import FWCore.ParameterSet.Config as cms

#------------------
# Hybrid clustering:
#------------------
# Producer for Box Particle Flow Super Clusters
from RecoEcal.EgammaClusterProducers.particleFlowSuperClusterECAL_cff import *
# Producer for energy corrections
#from RecoEcal.EgammaClusterProducers.correctedDynamicHybridSuperClusters_cfi import *
# PFECAL super clusters, either hybrid-clustering clone (Box) or mustache.
particleFlowSuperClusteringTask = cms.Task(particleFlowSuperClusterECAL)
particleFlowSuperClusteringSequence = cms.Sequence(particleFlowSuperClusteringTask)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
from RecoTICL.Superclustering.superclustering_cff import *

phase2_hgcal.toReplaceWith( particleFlowSuperClusteringTask, particleFlowSuperClusteringTask.copyAndAdd(particleFlowSuperClusterHGCal))
