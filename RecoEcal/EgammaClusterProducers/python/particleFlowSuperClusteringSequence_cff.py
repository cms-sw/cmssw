import FWCore.ParameterSet.Config as cms

#------------------
#Hybrid clustering:
#------------------
# Producer for Box Particle Flow Super Clusters
from RecoEcal.EgammaClusterProducers.particleFlowSuperClusterECAL_cfi import *
# Producer for energy corrections
#from RecoEcal.EgammaClusterProducers.correctedDynamicHybridSuperClusters_cfi import *
# PFECAL super clusters, either hybrid-clustering clone (Box) or mustache.
particleFlowSuperClusteringSequence = cms.Sequence(particleFlowSuperClusterECAL)

particleFlowSuperClusterHGCal = particleFlowSuperClusterECAL.clone()
_phase2_hgcal_particleFlowSuperClusteringSequence = particleFlowSuperClusteringSequence.copy()
_phase2_hgcal_particleFlowSuperClusteringSequence += particleFlowSuperClusterHGCal
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify(
    particleFlowSuperClusterHGCal,
    PFClusters = cms.InputTag('particleFlowClusterHGCal'),
    useRegression = cms.bool(False), #no HGCal regression yet
    use_preshower = cms.bool(False),
    PFBasicClusterCollectionEndcap = cms.string(""),   
    PFSuperClusterCollectionEndcap = cms.string(""),
    PFSuperClusterCollectionEndcapWithPreshower = cms.string(""),
    thresh_PFClusterEndcap = cms.double(1.5e-1) # 150 MeV threshold
)
phase2_hgcal.toReplaceWith( particleFlowSuperClusteringSequence, _phase2_hgcal_particleFlowSuperClusteringSequence )

