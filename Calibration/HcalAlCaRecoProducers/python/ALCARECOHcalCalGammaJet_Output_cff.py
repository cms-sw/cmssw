import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL Dijets
# output module 
#  module alcastreamHcalGammaJetOutput = PoolOutputModule
OutALCARECOHcalCalGammaJet_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalGammaJet')
    ),
    outputCommands = cms.untracked.vstring( 
                 'keep HcalNoiseSummary_hcalnoise_*_*',
                 #'keep recoPhotonCores_*_*_*',
                 'keep recoPhotonCores_gedPhotonCore_*_*',
                 'keep recoPhotonCores_photonCore_*_*',
                 'keep recoPhotonCores_reducedEgamma_reducedGedPhotonCores_*',
                 #'keep recoSuperClusters_*_*_*',
                 'keep recoSuperClusters_SCselector_*_*',
                 'keep recoSuperClusters_cleanedHybridSuperClusters_*_*',
                 'keep recoSuperClusters_correctedHybridSuperClusters_*_*',
                 'keep recoSuperClusters_correctedMulti5x5SuperClustersWithPreshower_*_*',
                 'keep recoSuperClusters_hfEMClusters_*_*',
                 'keep recoSuperClusters_hybridSuperClusters_*_*',
                 'keep recoSuperClusters_mergedSuperClusters__*',
                 'keep recoSuperClusters_multi5x5SuperClustersWithPreshower_*_*',
                 'keep recoSuperClusters_particleFlowEGamma_*_*',
                 'keep recoSuperClusters_uncleanedHybridSuperClusters_*_*',
                 'keep recoSuperClusters_uncleanedOnlyCorrectedHybridSuperClusters_*_*',
                 'keep recoSuperClusters_uncleanedOnlyCorrectedMulti5x5SuperClustersWithPreshower_*_*',
                 'keep recoSuperClusters_uncleanedOnlyMulti5x5SuperClustersWithPreshower_*_*',
                 'keep recoSuperClusters_multi5x5SuperClustersCleaned_*_*',
                 'keep recoSuperClusters_multi5x5SuperClustersUncleaned_*_*',
                 'keep recoSuperClusters_multi5x5SuperClusters_*_*',
                 'keep recoSuperClusters_particleFlowSuperClusterECAL_*_*',
                 'keep recoSuperClusters_reducedEgamma_reducedSuperClusters_*',
                 'keep recoTracks_generalTracks_*_*',
                 'keep *_particleFlow_*_*',
                 'keep recoPFBlocks_particleFlowBlock_*_*',
                 #'keep recoPFClusters_*_*_*',
                 'keep recoPFClusters_particleFlowClusterECAL_*_*',
                 'keep recoPFClusters_particleFlowClusterECALUncorrected_*_*',
                 'keep recoPFClusters_particleFlowClusterHBHE_*_*',
                 'keep recoPFClusters_particleFlowClusterHCAL_*_*',
                 'keep recoPFClusters_particleFlowClusterHF_*_*',
                 'keep recoPFClusters_particleFlowClusterHO_*_*',
                 'keep recoPFClusters_particleFlowClusterPS_*_*',
                 'keep *_GammaJetProd_*_*')
)

import copy
OutALCARECOHcalCalGammaJet=copy.deepcopy(OutALCARECOHcalCalGammaJet_noDrop)
OutALCARECOHcalCalGammaJet.outputCommands.insert(0, "drop *")
