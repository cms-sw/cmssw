import FWCore.ParameterSet.Config as cms

# output block for alcastream Electron
OutALCARECOEcalCalElectron_specific = cms.untracked.vstring(
    'drop reco*Clusters_hfEMClusters_*_*',
    'drop reco*Clusters_pfPhotonTranslator_*_*',
    'drop *EcalRecHit*_ecalRecHit_*_*',
    'drop *EcalrecHit*_*ecalPreshowerRecHit*_*EcalRecHitsES*_*',
    'drop *EcalRecHit*_reducedEcalRecHitsE*_*_*',
    'drop *_*Unclean*_*_*',
    'drop *_*unclean*_*_*',
    'drop *_*_*Unclean*_*',
    'drop *_*_*unclean*_*',
    'drop *CaloCluster*_*particleFlowEGamma*_*EBEEClusters*_*',
    'drop *CaloCluster*_*particleFlowEGamma*_*ESClusters*_*',
#    'keep *CaloCluster*_*alCaIsolatedElectrons*_*alcaCaloCluster*_*'
    'keep *CaloCluster*_alCaIsolatedElectrons_*alcaCaloCluster*_*'
)

OutALCARECOEcalCalElectron_noDrop = cms.PSet(
    # put this if you have a filter
    SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('pathALCARECOEcalCalZElectron', 'pathALCARECOEcalCalWElectron', 'pathALCARECOEcalCalZSCElectron')
    ),
    outputCommands = cms.untracked.vstring( 
    'keep *_pfMet_*_*', # met for Wenu selection
    #'keep *_kt6PFJetsForRhoCorrection_rho_*', #rho for effective area subtraction
    #'keep *_kt6PFJets_rho_*', #rho for effective area subtraction
    #'keep recoVertexs_offlinePrimaryVertices*_*_*',
    'keep recoVertexs_offlinePrimaryVertices_*_*',
    'keep recoVertexs_offlinePrimaryVerticesWithBS_*_*',
    'keep *BeamSpot_offlineBeamSpot_*_*',
    'keep *_allConversions_*_*',
    'keep *_conversions_*_*',
    #'keep *GsfTrack*_*_*_*',
    'keep *GsfTrack*_electronGsfTracks_*_*',
#    'keep *GsfTrack*_uncleanedOnlyElectronGsfTracks_*_*',
    'keep *_generator_*_*',
    'keep *_addPileupInfo_*_*',
    'keep *_genParticles_*_*',
    'keep recoGsfElectron*_gsfElectron*_*_*',
    #'keep recoGsfElectron*_gedGsfElectron*_*_*',
    'keep recoGsfElectron*_gedGsfElectrons_*_*',
#    'keep recoGsfElectron*_gedGsfElectronsTmp_*_*',
    'keep recoGsfElectron*_gedGsfElectronCores_*_*',
    'keep recoPhoton*_gedPhoton_*_*',
    #'keep recoCaloClusters_*_*_*',
    'keep recoCaloClusters_hfEMClusters_*_*',
    'keep recoCaloClusters_particleFlowEGamma_*_*',
    'keep recoCaloClusters_alCaIsolatedElectrons_*_*',
    'keep recoCaloClusters_cleanedHybridSuperClusters_*_*',
    'keep recoCaloClusters_hybridSuperClusters_*_*',
    'keep recoCaloClusters_uncleanedHybridSuperClusters_*_*',
    'keep recoCaloClusters_multi5x5BasicClustersCleaned_*_*',
    'keep recoCaloClusters_multi5x5BasicClustersUncleaned_*_*',
    'keep recoCaloClusters_multi5x5SuperClusters_*_*',
    'keep recoCaloClusters_particleFlowSuperClusterECAL_*_*',
    #'keep recoSuperClusters_*_*_*',
    'keep recoSuperClusters_SCselector_*_*',
    'keep recoSuperClusters_cleanedHybridSuperClusters_*_*',
    'keep recoSuperClusters_correctedHybridSuperClusters_*_*',
    'keep recoSuperClusters_correctedMulti5x5SuperClustersWithPreshower_*_*',
    'keep recoSuperClusters_hfEMClusters_*_*',
    'keep recoSuperClusters_hybridSuperClusters_*_*',
    'keep recoSuperClusters_mergedSuperClusters_*_*',
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
    #'keep recoPreshowerCluster*_*_*_*',
    'keep recoPreshowerCluster*_multi5x5SuperClustersWithPreshower_*_*',
    'keep recoPreshowerCluster*_uncleanedOnlyMulti5x5SuperClustersWithPreshower_*_*',
    'keep recoPreshowerCluster*_multi5x5PreshowerClusterShape_*_*',
    'keep *_pfElectronTranslator_*_*',
    #'keep *_*_*_HLT',
    #'keep *_generalTracks_*_*',
    #'keep reco*Track*Extra*_generalTracks_*_*',
    'keep *_alcaElectronTracksReducer_*_*',
    # for the trigger matching
    'keep *_l1extraParticles_*_*',
    'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
    'keep *_l1L1GtObjectMap_*_*',
    'keep edmConditionsInEventBlock_conditionsInEdm_*_*',
    'keep edmConditionsInLumiBlock_conditionsInEdm_*_*',
    'keep edmConditionsInRunBlock_conditionsInEdm_*_*',
    'keep *_TriggerResults_*_*',
    'keep *_hltTriggerSummaryAOD_*_HLT',
    # pfisolation CMSSW_5_3_X
    'keep *EcalRecHit*_alCaIsolatedElectrons_*_*',
    'keep *EcalRecHit*_reducedEcalRecHitsES_alCaRecHitsES_*',
    'keep *_fixedGridRhoFastjetAll_*_*'
    
    )
)

import copy
OutALCARECOEcalCalElectron=copy.deepcopy(OutALCARECOEcalCalElectron_noDrop)
OutALCARECOEcalCalElectron.outputCommands.insert(0, "drop *")
OutALCARECOEcalCalElectron.outputCommands+=OutALCARECOEcalCalElectron_specific

OutALCARECOEcalCalWElectron=copy.deepcopy(OutALCARECOEcalCalElectron)
OutALCARECOEcalCalWElectron_noDrop=copy.deepcopy(OutALCARECOEcalCalElectron_noDrop)
OutALCARECOEcalCalWElectron.SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('pathALCARECOEcalCalWElectron')   )
OutALCARECOEcalCalWElectron_noDrop.SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('pathALCARECOEcalCalWElectron')   )


OutALCARECOEcalCalZElectron=copy.deepcopy(OutALCARECOEcalCalElectron)
OutALCARECOEcalCalZElectron_noDrop=copy.deepcopy(OutALCARECOEcalCalElectron_noDrop)

OutALCARECOEcalCalZElectron.SelectEvents =  cms.untracked.PSet(
    SelectEvents = cms.vstring('pathALCARECOEcalCalZElectron', 'pathALCARECOEcalCalZSCElectron')    )
OutALCARECOEcalCalZElectron_noDrop.SelectEvents =  cms.untracked.PSet(
    SelectEvents = cms.vstring('pathALCARECOEcalCalZElectron', 'pathALCARECOEcalCalZSCElectron')    )



