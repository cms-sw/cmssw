
full_title = "RecoEcal collections (in RECO and AOD)"

full = {
    '0':['selectDigi', '*', 'No documentation'] ,
    '1':['reducedEcalRecHitsEB', 'edm::SortedCollection', 'Rechits from a 5x5 around Basic Clusters, for the ES, hits corresponding to clusters in EE'] ,
    '2':['reducedEcalRecHitsEE', 'edm::SortedCollection', 'Rechits from a 5x5 around Basic Clusters, for the ES, hits corresponding to clusters in EE'] ,
    '3':['reducedEcalRecHitsES', 'edm::SortedCollection', 'Rechits from a 5x5 around Basic Clusters, for the ES, hits corresponding to clusters in EE'] ,
    '4':['interestingEcalDetId*', '*', 'No documentation'] ,
    '5':['ecalWeightUncalibRecHit', '*', 'No documentation'] ,
    '6':['ecalPreshowerRecHit', '*', 'No documentation'] ,
    '7':['hybridSuperClusters', 'reco::BasicClusterCollection reco::ClusterShapeCollection reco::BasicClusterShapeAssociationCollection reco::SuperClusterCollection','Basic clusters, cluster shapes and super-clusters reconstructed with the hybrid algorithm with no energy corrections applied (barrel only)'] ,
    '8':['correctedHybridSuperClusters', 'reco::SuperClusterCollection', 'Super-clusters reconstructed with the hybrid algorithm with energy corrections applied (barrel only)'] ,
    '9':['multi5x5*', 'reco::BasicClusterCollection', 'Basic clusters in EE, 5x5 algorithm'] ,
    '10':['correctedMulti5x5*', 'reco::SuperCluster', 'Corrected superclusters in EE, 5x5 algorithm'] ,
    '11':['multi5x5SuperClustersWithPreshower', 'recoPreshowerClusters', 'Preshower clusters'] ,
    '12':['multi5x5PreshowerClusterShape', 'recoPreshowerClusterShapes', 'No documentation'] ,
    '13':['particleFlowSuperClusterECAL', '*', 'No documentation'],
    
     # Correction needed, because not matched with Event Content 
    '14':['hybridSuperClusters:uncleanOnlyHybridSuperClusters','reco::SuperClusterCollection','Only the SuperClusters containing anomalous signals, with no cleaning'] 
}

reco_title = "RecoEcal collections (in RECO only)"

reco = {
    '0':['selectDigi', '*', 'No documentation'] ,
    '1':['reducedEcalRecHitsEE', 'EcalRecHitsSorted', 'No documentation'] ,
    '2':['reducedEcalRecHitsEB', 'EcalRecHitsSorted', 'No documentation'] ,
    '3':['reducedEcalRecHitsES', 'EcalRecHitsSorted', 'No documentation'] ,
    '4':['hybridSuperClusters', '*', 'No documentation'] ,
    '5':['correctedHybridSuperClusters', 'recoSuperClusters', 'No documentation'] ,
    '6':['multi5x5SuperClusters', '*', 'No documentation'] ,
    '7':['multi5x5SuperClusters', 'recoSuperClusters', 'No documentation'] ,
    '8':['multi5x5SuperClustersWithPreshower', 'recoSuperClusters', 'No documentation'] ,
    '9':['correctedMulti5x5SuperClustersWithPreshower', 'recoSuperClusters', 'No documentation'] ,
    '10':['multi5x5SuperClustersWithPreshower', 'recoPreshowerClusters', 'No documentation'] ,
    '11':['multi5x5PreshowerClusterShape', 'recoPreshowerClusterShapes', 'No documentation'] ,
    '12':['particleFlowSuperClusterECAL', '*', 'No documentation'] 
}

aod_title = "RecoEcal collections (in AOD only)"

aod = {
    '0':['selectDigi', '*', 'No documentation'] ,
    '1':['reducedEcalRecHitsEB', 'EcalRecHitsSorted', 'No documentation'] ,
    '2':['reducedEcalRecHitsEE', 'EcalRecHitsSorted', 'No documentation'] ,
    '3':['reducedEcalRecHitsES', 'EcalRecHitsSorted', 'No documentation'] 
}