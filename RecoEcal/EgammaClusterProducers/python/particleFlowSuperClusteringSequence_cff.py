import FWCore.ParameterSet.Config as cms

#------------------
#Hybrid clustering:
#------------------
# Producer for Box Particle Flow Super Clusters
from RecoEcal.EgammaClusterProducers.particleFlowSuperClusterECAL_cff import *
# Producer for energy corrections
#from RecoEcal.EgammaClusterProducers.correctedDynamicHybridSuperClusters_cfi import *
# PFECAL super clusters, either hybrid-clustering clone (Box) or mustache.
particleFlowSuperClusteringTask = cms.Task(particleFlowSuperClusterECAL)
particleFlowSuperClusteringSequence = cms.Sequence(particleFlowSuperClusteringTask)

particleFlowSuperClusterHGCal = particleFlowSuperClusterECAL.clone()
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify(
    particleFlowSuperClusterHGCal,
    PFClusters = cms.InputTag('particleFlowClusterHGCal'),
    useRegression = cms.bool(False), #no HGCal regression yet
    use_preshower = cms.bool(False),
    PFBasicClusterCollectionEndcap = cms.string(""),
    PFSuperClusterCollectionEndcap = cms.string(""),
    PFSuperClusterCollectionEndcapWithPreshower = cms.string(""),
    thresh_PFClusterEndcap = cms.double(1.5e-1), # 150 MeV threshold
    dropUnseedable = cms.bool(True),
)

particleFlowSuperClusterHGCalFromMultiCl = particleFlowSuperClusterHGCal.clone()
phase2_hgcal.toModify(
    particleFlowSuperClusterHGCalFromMultiCl,
    PFClusters = cms.InputTag('particleFlowClusterHGCalFromMultiCl')
)
_phase2_hgcal_particleFlowSuperClusteringTask = particleFlowSuperClusteringTask.copy()
_phase2_hgcal_particleFlowSuperClusteringTask.add(particleFlowSuperClusterHGCal)
_phase2_hgcal_particleFlowSuperClusteringTask.add(particleFlowSuperClusterHGCalFromMultiCl)

phase2_hgcal.toReplaceWith( particleFlowSuperClusteringTask, _phase2_hgcal_particleFlowSuperClusteringTask )

