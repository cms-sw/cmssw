import FWCore.ParameterSet.Config as cms

# AOD content
RecoParticleFlowAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep recoPFRecHits_particleFlowClusterECAL_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterHCAL_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterHO_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterHF_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterPS_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitECAL_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitHBHE_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitHF_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitHO_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitPS_Cleaned_*',
    'keep recoCaloClusters_particleFlowEGamma_*_*',
    'keep recoSuperClusters_particleFlowEGamma_*_*',
    'keep recoCaloClusters_particleFlowSuperClusterECAL_*_*',
    'keep recoSuperClusters_particleFlowSuperClusterECAL_*_*',
    'keep recoConversions_particleFlowEGamma_*_*',
    'keep recoPFCandidates_particleFlow_*_*',
    'keep recoPFCandidates_particleFlowTmp_AddedMuonsAndHadrons_*',
    'keep recoPFCandidates_particleFlowTmp_CleanedCosmicsMuons_*',
    'keep recoPFCandidates_particleFlowTmp_CleanedFakeMuons_*',
    'keep recoPFCandidates_particleFlowTmp_CleanedHF_*',
    'keep recoPFCandidates_particleFlowTmp_CleanedPunchThroughMuons_*',
    'keep recoPFCandidates_particleFlowTmp_CleanedPunchThroughNeutralHadrons_*',
    'keep recoPFCandidates_particleFlowTmp_CleanedTrackerAndGlobalMuons_*',
    'keep *_particleFlow_electrons_*',
    'keep *_particleFlow_photons_*',
    'keep *_particleFlow_muons_*',
    'keep recoCaloClusters_pfElectronTranslator_*_*',
    'keep recoPreshowerClusters_pfElectronTranslator_*_*',
    'keep recoSuperClusters_pfElectronTranslator_*_*',
    'keep recoCaloClusters_pfPhotonTranslator_*_*',
    'keep recoPreshowerClusters_pfPhotonTranslator_*_*',
    'keep recoSuperClusters_pfPhotonTranslator_*_*',
    'keep recoPhotons_pfPhotonTranslator_*_*',
    'keep recoPhotonCores_pfPhotonTranslator_*_*',
    'keep recoConversions_pfPhotonTranslator_*_*',
    'keep *_particleFlowPtrs_*_*',
    'keep *_particleFlowTmpPtrs_*_*',
    'keep *_chargedHadronPFTrackIsolation_*_*')
)

# mods for HGCAL and timing
# Some SC content also defined in RecoEcal/Configuration/python/RecoEcal_EventContent_cff.py
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_hgcal.toModify( RecoParticleFlowAOD,  
    outputCommands = RecoParticleFlowAOD.outputCommands + ['keep recoPFRecHits_particleFlowRecHitHGC_Cleaned_*',
                                                           'keep recoPFClusters_particleFlowClusterHGCal__*',
                                                           'keep recoPFClusters_particleFlowClusterHGCalFromMultiCl__*',
                                                           'keep recoSuperClusters_simPFProducer_*_*'])
phase2_timing.toModify( RecoParticleFlowAOD,
    outputCommands = RecoParticleFlowAOD.outputCommands + ['keep *_ecalBarrelClusterFastTimer_*_*'])

# RECO content
RecoParticleFlowRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep recoPFClusters_particleFlowClusterECAL_*_*',
    'keep recoPFClusters_particleFlowClusterHCAL_*_*',
    'keep recoPFClusters_particleFlowClusterHO_*_*',
    'keep recoPFClusters_particleFlowClusterHF_*_*',
    'keep recoPFClusters_particleFlowClusterPS_*_*',
    'keep recoPFBlocks_particleFlowBlock_*_*',
    'keep recoPFCandidates_particleFlowEGamma_*_*',
    'keep recoPFCandidates_particleFlowTmp_electrons_*',
    'keep recoPFDisplacedVertexs_particleFlowDisplacedVertex_*_*',
    'keep *_pfElectronTranslator_*_*',
    'keep *_pfPhotonTranslator_*_*',
    'keep *_trackerDrivenElectronSeeds_preid_*')
)
RecoParticleFlowRECO.outputCommands.extend(RecoParticleFlowAOD.outputCommands)

phase2_hgcal.toModify( RecoParticleFlowRECO,
    outputCommands = RecoParticleFlowRECO.outputCommands + ['keep *_particleFlowSuperClusterHGCalFromMultiCl_*_*',
                                                            'keep recoPFBlocks_simPFProducer_*_*'])

# Full Event content
RecoParticleFlowFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()                                                                 
)
RecoParticleFlowFEVT.outputCommands.extend(RecoParticleFlowRECO.outputCommands)
    
phase2_hgcal.toModify( RecoParticleFlowFEVT, 
    outputCommands = RecoParticleFlowFEVT.outputCommands + ['keep recoPFRecHits_particleFlowClusterECAL__*',
                                                            'keep recoPFRecHits_particleFlowRecHitHGC__*',
                                                            'keep *_simPFProducer_*_*'])

from Configuration.ProcessModifiers.mlpf_cff import mlpf
from RecoParticleFlow.PFProducer.mlpf_EventContent_cff import MLPF_RECO

mlpf.toModify(RecoParticleFlowRECO,
    outputCommands = RecoParticleFlowRECO.outputCommands + MLPF_RECO.outputCommands)
