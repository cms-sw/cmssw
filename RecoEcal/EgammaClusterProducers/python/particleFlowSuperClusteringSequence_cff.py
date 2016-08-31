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

from Configuration.StandardSequences.Eras import eras
particleFlowSuperClusterHGCal = particleFlowSuperClusterECAL.clone()
_phase2_hgcal_particleFlowSuperClusteringSequence = particleFlowSuperClusteringSequence.copy()
_phase2_hgcal_particleFlowSuperClusteringSequence += particleFlowSuperClusterHGCal
eras.phase2_hgcal.toModify(
    particleFlowSuperClusterHGCal,
    PFClusters = cms.InputTag('particleFlowClusterHGCal'),
    useRegression = cms.bool(False), #no HGCal regression yet
    use_preshower = cms.bool(False),
    PFBasicClusterCollectionEndcap = cms.string(""),   
    PFSuperClusterCollectionEndcap = cms.string(""),
    PFSuperClusterCollectionEndcapWithPreshower = cms.string(""),
    thresh_PFClusterEndcap = cms.double(1.5e-1) # 150 MeV threshold
)
eras.phase2_hgcal.toReplaceWith( particleFlowSuperClusteringSequence, _phase2_hgcal_particleFlowSuperClusteringSequence )

