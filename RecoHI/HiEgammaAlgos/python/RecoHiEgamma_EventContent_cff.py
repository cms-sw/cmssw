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
        'keep recoSuperClusters_cleanedHybridSuperClusters_*_*',
        'keep recoSuperClusters_correctedEndcapSuperClustersWithPreshower_*_*', 
        'keep recoSuperClusters_correctedHybridSuperClusters_*_*',
        'keep recoSuperClusters_correctedMulti5x5SuperClustersWithPreshower_*_*',
        'keep recoSuperClusters_hybridSuperClusters_*_*',
        'keep recoSuperClusters_islandSuperClusters_*_*',
        'keep recoSuperClusters_mergedSuperClusters_*_*',
        'keep recoSuperClusters_multi5x5SuperClusters_*_*',
        'keep recoSuperClusters_multi5x5SuperClustersCleaned_*_*',
        'keep recoSuperClusters_multi5x5SuperClustersUncleaned_*_*',
        'keep recoSuperClusters_multi5x5SuperClustersWithPreshower_*_*',
        'keep recoSuperClusters_particleFlowEGamma_*_*',
        'keep recoSuperClusters_particleFlowSuperClusterECAL_*_*',
        'keep recoSuperClusters_uncleanedHybridSuperClusters_*_*',
        'keep recoSuperClusters_uncleanedOnlyCorrectedHybridSuperClusters_*_*',
        'keep recoSuperClusters_uncleanedOnlyCorrectedMulti5x5SuperClustersWithPreshower_*_*',
        'keep recoSuperClusters_uncleanedOnlyMulti5x5SuperClustersWithPreshower_*_*',
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
        'keep EcalRecHitsSorted_ecalRecHit_*_*',
        'keep EcalRecHitsSorted_ecalPreshowerRecHit_*_*',
        'keep EBSrFlagsSorted_ecalDigis__*',
        'keep EESrFlagsSorted_ecalDigis__*',
        'keep recoPFCandidates_particleFlowEGamma_*_*',
        'keep recoPFCandidates_particleFlowTmp_*_*',
        'keep recoGsfElectrons_ecalDrivenGsfElectrons_*_*',
        'keep recoGsfElectrons_electronsWithPresel_*_*',
        'keep recoGsfElectrons_gedGsfElectronsTmp_*_*',
        'keep recoGsfElectrons_mvaElectrons_*_*')
)
RecoHiEgammaRECO.outputCommands.extend(RecoHiEgammaAOD.outputCommands)

# FEVT content
RecoHiEgammaFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep recoSuperClusters_*_*_*',
        'keep recoCaloClusters_*_*_*',
        'keep EcalRecHitsSorted_*_*_*',
        'keep recoPFCandidates_*_*_*',
        'keep recoElectronSeeds_*_*_*',
        'keep recoGsfElectrons_*_*_*')
)
RecoHiEgammaFEVT.outputCommands.extend(RecoHiEgammaRECO.outputCommands)
