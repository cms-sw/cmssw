import FWCore.ParameterSet.Config as cms

# output block for alcastream Electron
OutALCARECOEcalCalElectron_specific = cms.untracked.vstring(
    'drop reco*Clusters_hfEMClusters_*_*',
    'drop reco*Clusters_pfPhotonTranslator_*_*',
    'drop *EcalRecHit*_ecalRecHit_*_*',
    'drop *EcalrecHit*_*ecalPreshowerRecHit*_*EcalRecHitsES*_*',
    'drop *EcalRecHit*_reducedEcalRecHitsE*_*_*',
    'drop *_*Cleaned_*_*',
    'drop *_*cleaned*_*_*',
    'drop *_*Unclean*_*_*',
    'drop *_*unclean*_*_*',
    'drop *_*_*Unclean*_*',
    'drop *_*_*unclean*_*',
    'keep recoSuperClusters_*_uncleanOnly_*',  # kept for safety
    'drop *_*_multi5x5Barrel*Clusters_*',
    'drop *CaloCluster*_*particleFlowEGamma*_*EBEEClusters*_*',
    'drop *CaloCluster*_*particleFlowEGamma*_*ESClusters*_*',
    'drop *_dqmL1ExtraParticles_*_*',
    'drop recoSuperClusters_mergedSuperClusters_*_*',
    'keep *CaloCluster*_*alCaIsolatedElectrons*_*alcaCaloCluster*_*'    
)

OutALCARECOEcalCalElectron_noDrop = cms.PSet(
    # put this if you have a filter
    SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('pathALCARECOEcalCalZElectron', 'pathALCARECOEcalCalWElectron', 'pathALCARECOEcalCalZSCElectron')
    ),
    outputCommands = cms.untracked.vstring( 
    'keep *_pfMet_*_*', # met for Wenu selection
    'keep *_kt6PFJetsForRhoCorrection_rho_*', #rho for effective area subtraction
    'keep *_kt6PFJets_rho_*', #rho for effective area subtraction
    'keep recoVertexs_offlinePrimaryVertices*_*_*',
    'keep *BeamSpot_offlineBeamSpot_*_*',
    'keep *_allConversions_*_*',
    'keep *_conversions_*_*',
    'keep *GsfTrack*_*_*_*',
    'keep *_generator_*_*',
    'keep *_addPileupInfo_*_*',
    'keep *_genParticles_*_*',
    'keep recoGsfElectron*_gsfElectrons*_*_*',
    'keep recoGsfElectron*_gedGsfElectrons_*_*',
    'keep recoPhoton*_gedPhotons_*_*',
    'keep recoCaloClusters_*_*_*',
    'keep recoSuperClusters_*_*_*',
    'keep recoPreshowerCluster*_*_*_*',
    'keep *_pfElectronTranslator_*_*',
    #'keep *_*_*_HLT',
    #'keep *_generalTracks_*_*',
    #'keep reco*Track*Extra*_generalTracks_*_*',
    'keep *_alcaElectronTracksReducer_*_*',
    # for the trigger matching
    #'keep *_l1extraParticles_*_*',
    'keep l1extraL1EmParticles_*_*_*',
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
    )
)

import copy
OutALCARECOEcalCalElectron=copy.deepcopy(OutALCARECOEcalCalElectron_noDrop)
OutALCARECOEcalCalElectron.outputCommands.insert(0, "drop *")
OutALCARECOEcalCalElectron.outputCommands+=OutALCARECOEcalCalElectron_specific

OutALCARECOEcalCalWElectron=copy.deepcopy(OutALCARECOEcalCalElectron)
OutALCARECOEcalCalWElectron_noDrop=copy.deepcopy(OutALCARECOEcalCalElectron_noDrop)
OutALCARECOEcalCalZElectron=copy.deepcopy(OutALCARECOEcalCalElectron)
OutALCARECOEcalCalZElectron_noDrop=copy.deepcopy(OutALCARECOEcalCalElectron_noDrop)




