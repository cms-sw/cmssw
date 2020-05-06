import FWCore.ParameterSet.Config as cms

# AOD content 
RecoHiEgammaAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep floatedmValueMap_*_*_*',
        'keep recoGsfElectrons_gedGsfElectronsTmp_*_*',
        'keep recoSuperClusters_correctedIslandBarrelSuperClusters_*_*',
        'keep recoSuperClusters_correctedIslandEndcapSuperClusters_*_*',
        'keep recoPhotons_gedPhotonsTmp_*_*',
        'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducer_*_*',
        'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerGED_*_*',
        'keep recoElectronSeeds_ecalDrivenElectronSeeds_*_*',
        'keep recoTrackExtras_electronGsfTracks_*_*')
)

# RECO content
RecoHiEgammaRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep recoSuperClusters_correctedHybridSuperClusters_*_*',        
        'keep recoSuperClusters_correctedMulti5x5SuperClustersWithPreshower_*_*',
        'keep recoSuperClusters_hfEMClusters_*_*',
        'keep recoSuperClusters_hybridSuperClusters_*_*',
        'keep recoSuperClusters_multi5x5SuperClusters_*_*',
        'keep recoSuperClusters_multi5x5SuperClustersWithPreshower_*_*',
        'keep recoSuperClusters_particleFlowEGamma_*_*',
        'keep recoSuperClusters_particleFlowSuperClusterECAL_*_*',
        'keep recoSuperClusters_particleFlowSuperClusterOOTECAL_*_*',
        'keep recoSuperClusters_lowPtGsfElectronSuperClusters_*_*',
        'keep recoCaloClusters_hfEMClusters_*_*',
        'keep recoCaloClusters_particleFlowEGamma_*_*',
        'keep recoCaloClusters_cleanedHybridSuperClusters_*_*',
        'keep recoCaloClusters_hybridSuperClusters_*_*',
        'keep recoCaloClusters_uncleanedHybridSuperClusters_*_*',
        'keep recoCaloClusters_islandBasicClusters_*_*',
        'keep recoCaloClusters_multi5x5BasicClustersCleaned_*_*',
        'keep recoCaloClusters_multi5x5BasicClustersUncleaned_*_*',
        'keep recoCaloClusters_multi5x5SuperClusters_*_*',
        'keep recoCaloClusters_particleFlowSuperClusterECAL_*_*',
        'keep recoCaloClusters_particleFlowSuperClusterOOTECAL_*_*',
        'keep recoCaloClusters_lowPtGsfElectronSuperClusters_*_*',
        'keep recoCaloClusters_pfElectronTranslator_*_*',
        'keep recoCaloClusters_pfPhotonTranslator_*_*',
        'keep EcalRecHitsSorted_ecalRecHit_*_*',
        'keep EcalRecHitsSorted_ecalPreshowerRecHit_*_*',
        'keep EBSrFlagsSorted_ecalDigis__*',
        'keep EESrFlagsSorted_ecalDigis__*',
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
        'keep recoPFCandidates_particleFlowEGamma_*_*',
        'keep recoPFCandidates_particleFlowTmp_*_*',
        "keep recoGsfElectrons_ecalDrivenGsfElectrons_*_*",
        "keep recoGsfElectrons_electronsWithPresel_*_*",
        "keep recoGsfElectrons_gedGsfElectronsTmp_*_*",
        "keep recoGsfElectrons_mvaElectrons_*_*")
)
RecoHiEgammaRECO.outputCommands.extend(RecoHiEgammaAOD.outputCommands)

# FEVT content
RecoHiEgammaFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep recoSuperClusters_*_*_*',
        'keep recoCaloClusters_*_*_*',
        'keep EcalRecHitsSorted_*_*_*',
        'keep floatedmValueMap_*_*_*',
        'keep recoPFCandidates_*_*_*',
        "keep recoElectronSeeds_*_*_*",
        "keep recoGsfElectrons_*_*_*")
)
RecoHiEgammaFEVT.outputCommands.extend(RecoHiEgammaRECO.outputCommands)
