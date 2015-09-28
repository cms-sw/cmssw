import FWCore.ParameterSet.Config as cms


OutALCARECOEcalCalElectron_noDrop = cms.PSet(
    # put this if you have a filter
    SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('pathALCARECOEcalCalZElectron', 'pathALCARECOEcalCalWElectron', 'pathALCARECOEcalCalZSCElectron')
    ),
    outputCommands = cms.untracked.vstring( 
    'keep *_pfMet_*_*',
    'keep recoPFMETs_pfMet_*_*', # met for Wenu selection

    'keep *_kt6PFJetsForRhoCorrection_rho_*', #rho for effective area subtraction
    'keep *_kt6PFJets_rho_*', #rho for effective area subtraction
    'keep *_addPileupInfo_*_*',

    'keep *_generator_*_*',
    'keep *_genParticles_*_*',

    'keep recoBeamSpot_offlineBeamSpot_*_*',
    'keep recoVertexs_offlinePrimaryVerticesWithBS_*_*',
    'keep recoVertexs_offlinePrimaryVertices_*_*',
    'keep recoGsfTracks_electronGsfTracks_*_*',
    'keep *GsfTrack*_*_*_*', # gsfTrackExtra
    'keep *_alcaElectronTracksReducer_*_*',

    'keep *_allConversions_*_*',
    'keep *_conversions_*_*',

    'keep recoGsfElectrons_gsfElectrons_*_*',
    'keep recoGsfElectronCores_gsfElectronCores_*_*',
    'keep recoGsfElectrons_gedGsfElectrons_*_*',
    'keep recoGsfElectronCores_gedGsfElectronCores_*_*',

    'keep recoPhotons_gedPhotons_*_*',

    'keep recoConversions_allConversions_*_*',
    'keep recoConversions_conversions_*_*',

    'keep *EcalRecHit*_alCaIsolatedElectrons_*_*',
    'keep *EcalRecHit*_reducedEcalRecHitsES_alCaRecHitsES_*',

    'keep recoCaloClusters_alCaIsolatedElectrons_alcaCaloCluster_*', ## producer to be fixed
    'keep recoCaloClusters_hybridSuperClusters_hybridBarrelBasicClusters_*',
    'keep recoCaloClusters_multi5x5SuperClusters_multi5x5EndcapBasicClusters_*',
    'keep recoCaloClusters_particleFlowSuperClusterECAL_particleFlowBasicClusterECALEndcap_*',
    'keep recoCaloClusters_particleFlowSuperClusterECAL_particleFlowBasicClusterECALBarrel_*',
    'keep recoCaloClusters_particleFlowSuperClusterECAL_particleFlowBasicClusterECALPreshower_*',
    'keep recoCaloClusters_particleFlowEGamma_EBEEClusters_*',
    'keep recoCaloClusters_particleFlowEGamma_ESClusters_*',

    'keep recoSuperClusters_correctedHybridSuperClusters_*_*',
    'keep recoSuperClusters_correctedMulti5x5SuperClustersWithPreshower_*_*',
    'keep recoSuperClusters_particleFlowSuperClusterECAL_particleFlowSuperClusterECALBarrel_*',
    'keep recoSuperClusters_particleFlowSuperClusterECAL_particleFlowSuperClusterECALEndcapWithPreshower_*',
    'keep recoSuperClusters_particleFlowEGamma_*_*',
    'keep recoSuperClusters_SCselector_*_*',

    'keep recoPreshowerClusters_multi5x5SuperClustersWithPreshower_preshowerYClusters_*',
    'keep recoPreshowerClusters_multi5x5SuperClustersWithPreshower_preshowerXClusters_*',
    'keep recoPreshowerClusterShapes_multi5x5PreshowerClusterShape_multi5x5PreshowerYClustersShape_*',
    'keep recoPreshowerClusterShapes_multi5x5PreshowerClusterShape_multi5x5PreshowerXClustersShape_*',

    'keep *_pfElectronTranslator_*_*',

    # for the trigger matching
    'keep L1GlobalTriggerObjectMaps_l1L1GtObjectMap_*_*',
    'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
    'keep l1extraL1EmParticles_l1extraParticles_*_*',
    'keep *_l1L1GtObjectMap_*_*',
    'keep edmConditionsInEventBlock_conditionsInEdm_*_*',
    'keep edmConditionsInLumiBlock_conditionsInEdm_*_*',
    'keep edmConditionsInRunBlock_conditionsInEdm_*_*',
    'keep edmTriggerResults_TriggerResults_*_*',
    'keep triggerTriggerEvent_hltTriggerSummaryAOD_*_*',
    )
)

import copy
OutALCARECOEcalCalElectron=copy.deepcopy(OutALCARECOEcalCalElectron_noDrop)
OutALCARECOEcalCalElectron.outputCommands.insert(0, "drop *")

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



