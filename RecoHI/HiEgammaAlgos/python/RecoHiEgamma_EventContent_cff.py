import FWCore.ParameterSet.Config as cms

RecoHiEgammaFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep recoSuperClusters_*_*_*',
    'keep recoCaloClusters_*_*_*',
    'keep EcalRecHitsSorted_*_*_*',
    'keep floatedmValueMap_*_*_*',
    'keep recoPFCandidates_*_*_*',
    "drop recoPFClusters_*_*_*",
    "keep recoElectronSeeds_*_*_*",
    "keep recoGsfElectrons_*_*_*",
    'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducer_*_*',
    'keep recoPhotons_gedPhotonsTmp_*_*',
    'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerGED_*_*',
    'keep recoElectronSeeds_ecalDrivenElectronSeeds_*_*',
    'keep recoTrackExtras_electronGsfTracks_*_*'
    )
    )

RecoHiEgammaRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
    #'keep recoSuperClusters_*_*_*',
     'keep recoSuperCluster_cleanedHybridSuperClusters_*_*',
     'keep recoSuperCluster_correctedEndcapSuperClustersWithPreshower_*_*',     
     'keep recoSuperCluster_correctedHybridSuperClusters_*_*',        
     'keep recoSuperCluster_correctedIslandBarrelSuperClusters_*_*',
     'keep recoSuperCluster_correctedIslandEndcapSuperClusters_*_*',
     'keep recoSuperCluster_correctedMulti5x5SuperClustersWithPreshower_*_*',
     'keep recoSuperCluster_hybridSuperClusters_*_*',
     'keep recoSuperCluster_islandSuperClusters_*_*',
     'keep recoSuperCluster_mergedSuperClusters_*_*',
     'keep recoSuperCluster_multi5x5SuperClusters_*_*',
     'keep recoSuperCluster_multi5x5SuperClustersCleaned_*_*',
     'keep recoSuperCluster_multi5x5SuperClustersUncleaned_*_*',
     'keep recoSuperCluster_multi5x5SuperClustersWithPreshower_*_*',
     'keep recoSuperCluster_particleFlowEGamma_*_*',
     'keep recoSuperCluster_particleFlowSuperClusterECAL_*_*',
     'keep recoSuperCluster_uncleanedHybridSuperClusters_*_*',
     'keep recoSuperCluster_uncleanedOnlyCorrectedHybridSuperClusters_*_*',
     'keep recoSuperCluster_uncleanedOnlyCorrectedMulti5x5SuperClustersWithPreshower_*_*',
     'keep recoSuperCluster_uncleanedOnlyMulti5x5SuperClustersWithPreshower_*_*',
    #'keep recoCaloClusters_*_*_*',
     'keep recoCaloClusters_particleFlowEGamma_*_*',
     'keep recoCaloClusters_cleanedHybridSuperClusters_*_*',
     'keep recoCaloClusters_hybridSuperClusters_*_*',
     'keep recoCaloClusters_uncleanedHybridSuperClusters_*_*',
     'keep recoCaloClusters_islandBasicClusters_*_*',
     'keep recoCaloClusters_multi5x5BasicClustersCleaned_*_*',
     'keep recoCaloClusters_multi5x5BasicClustersUncleaned_*_*',
     'keep recoCaloClusters_multi5x5SuperClusters_*_*',
     'keep recoCaloClusters_particleFlowSuperClusterECAL_*_*',
     'keep recoCaloClusters_multi5x5SuperClusters_*_*',
    #'keep EcalRecHitsSorted_*_*_*',
    'keep EcalRecHitsSorted_ecalRecHit_*_*',
    'keep EcalRecHitsSorted_ecalPreshowerRecHit_*_*',
    #'keep floatedmValueMap_*_*_*',  # isolation not created yet in RECO step, but in case it is later
    'keep floatedmValueMap_hiDetachedTripletStepQual_MVAVals_*',        
    'keep floatedmValueMap_hiDetachedTripletStepSelector_MVAVals_*',    
    'keep floatedmValueMap_hiGeneralTracks_MVAVals_*',                  
    'keep floatedmValueMap_hiInitialStepSelector_MVAVals_*',            
    'keep floatedmValueMap_hiLowPtTripletStepQual_MVAVals_*',           
    'keep floatedmValueMap_hiLowPtTripletStepSelector_MVAVals_*', 
    'keep floatedmValueMap_hiPixelPairStepSelector_MVAVals_*',          
    'keep floatedmValueMap_hiRegitMuInitialStepSelector_MVAVals_*', 
    'keep floatedmValueMap_hiRegitMuMixedTripletStepSelector_MVAVals_*',
    'keep floatedmValueMap_hiRegitMuPixelLessStepSelector_MVAVals_*', 
    'keep floatedmValueMap_hiRegitMuPixelPairStepSelector_MVAVals_*', 
    #'keep recoPFCandidates_*_*_*',
    'keep recoPFCandidates_particleFlowEGamma_*_*',
    'keep recoPFCandidates_particleFlowTmp_*_*',
    "drop recoPFClusters_*_*_*",
    "keep recoElectronSeeds_*_*_*",
    "keep recoGsfElectrons_*_*_*",
    'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducer_*_*',
    'keep recoPhotons_gedPhotonsTmp_*_*',
    'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerGED_*_*',
    'keep recoElectronSeeds_ecalDrivenElectronSeeds_*_*',
     'keep recoTrackExtras_electronGsfTracks_*_*'
    )
    )

RecoHiEgammaAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep floatedmValueMap_*_*_*',
    'keep recoGsfElectrons_gedGsfElectronsTmp_*_*',
    'keep recoSuperClusters_correctedIslandBarrelSuperClusters_*_*',
    'keep recoSuperClusters_correctedIslandEndcapSuperClusters_*_*',
    'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducer_*_*',
    'keep recoPhotons_gedPhotonsTmp_*_*',
    'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerGED_*_*',
    'keep recoElectronSeeds_ecalDrivenElectronSeeds_*_*',
    'keep recoTrackExtras_electronGsfTracks_*_*'
    )
    )
