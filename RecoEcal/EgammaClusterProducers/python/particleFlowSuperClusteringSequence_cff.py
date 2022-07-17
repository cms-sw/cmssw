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
    PFClusters                     = 'particleFlowClusterHGCal',
    useRegression                  = True,
    PFBasicClusterCollectionEndcap = "",
    PFSuperClusterCollectionEndcap = "",
    PFSuperClusterCollectionEndcapWithPreshower = "",
    thresh_PFClusterEndcap         = 1.5e-1, # 150 MeV threshold
    dropUnseedable                 = True,
)

phase2_hgcal.toModify( particleFlowSuperClusterHGCal.regressionConfig,
    regressionKeyEE = "superclus_hgcal_mean_offline",
    uncertaintyKeyEE = "superclus_hgcal_sigma_offline",
    isPhaseII = True,
    hgcalRecHits = "particleFlowRecHitHGC"
            
)
_phase2_hgcal_particleFlowSuperClusteringTask = particleFlowSuperClusteringTask.copy()
_phase2_hgcal_particleFlowSuperClusteringTask.add(particleFlowSuperClusterHGCal)

phase2_hgcal.toReplaceWith( particleFlowSuperClusteringTask, _phase2_hgcal_particleFlowSuperClusteringTask )
