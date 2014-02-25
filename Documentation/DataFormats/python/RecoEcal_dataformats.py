'''
    Created on Jun 26, 2013 
    @author:  Mantas Stankevicius
    @contact: mantas.stankevicius@cern.ch
    http://cmsdoxy.web.cern.ch/cmsdoxy/dataformats/
    
    @responsible: 
    
'''

json = {
  "full": {
    "title": "RecoEcal collections (in RECO and AOD)",
    "data": [
     {
      "instance": "multi5x5SuperClustersWithPreshower",
      "container": "recoPreshowerClusters",
      "desc": "Preshower clusters"
     },
     {
      "instance": "correctedMulti5x5*",
      "container": "reco::SuperCluster",
      "desc": "Corrected superclusters in EE, 5x5 algorithm"
     },
     {
      "instance": "particleFlowSuperClusterECAL",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "multi5x5PreshowerClusterShape",
      "container": "recoPreshowerClusterShapes",
      "desc": "No documentation"
     },
     {
      "instance": "hybridSuperClusters:uncleanOnlyHybridSuperClusters",
      "container": "reco::SuperClusterCollection",
      "desc": "Only the SuperClusters containing anomalous signals, with no cleaning"
     },
     {
      "instance": "reducedEcalRecHitsEB",
      "container": "edm::SortedCollection",
      "desc": "Rechits from a 5x5 around Basic Clusters, for the ES, hits corresponding to clusters in EE"
     },
     {
      "instance": "selectDigi",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "reducedEcalRecHitsES",
      "container": "edm::SortedCollection",
      "desc": "Rechits from a 5x5 around Basic Clusters, for the ES, hits corresponding to clusters in EE"
     },
     {
      "instance": "reducedEcalRecHitsEE",
      "container": "edm::SortedCollection",
      "desc": "Rechits from a 5x5 around Basic Clusters, for the ES, hits corresponding to clusters in EE"
     },
     {
      "instance": "ecalWeightUncalibRecHit",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "interestingEcalDetId*",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "hybridSuperClusters",
      "container": "reco::BasicClusterCollection reco::ClusterShapeCollection reco::BasicClusterShapeAssociationCollection reco::SuperClusterCollection",
      "desc": "Basic clusters, cluster shapes and super-clusters reconstructed with the hybrid algorithm with no energy corrections applied (barrel only)"
     },
     {
      "instance": "ecalPreshowerRecHit",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "multi5x5*",
      "container": "reco::BasicClusterCollection",
      "desc": "Basic clusters in EE, 5x5 algorithm"
     },
     {
      "instance": "correctedHybridSuperClusters",
      "container": "reco::SuperClusterCollection",
      "desc": "Super-clusters reconstructed with the hybrid algorithm with energy corrections applied (barrel only)"
     }
    ]
  },
  "aod": {
    "title": "RecoEcal collections (in AOD only)",
    "data": [
     {
      "instance": "reducedEcalRecHitsEB",
      "container": "EcalRecHitsSorted",
      "desc": "No documentation"
     },
     {
      "instance": "selectDigi",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "reducedEcalRecHitsES",
      "container": "EcalRecHitsSorted",
      "desc": "No documentation"
     },
     {
      "instance": "reducedEcalRecHitsEE",
      "container": "EcalRecHitsSorted",
      "desc": "No documentation"
     }
    ]
  },
  "reco": {
    "title": "RecoEcal collections (in RECO only)",
    "data": [
     {
      "instance": "multi5x5PreshowerClusterShape",
      "container": "recoPreshowerClusterShapes",
      "desc": "No documentation"
     },
     {
      "instance": "multi5x5SuperClustersWithPreshower",
      "container": "recoPreshowerClusters",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowSuperClusterECAL",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "reducedEcalRecHitsEE",
      "container": "EcalRecHitsSorted",
      "desc": "No documentation"
     },
     {
      "instance": "selectDigi",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "reducedEcalRecHitsES",
      "container": "EcalRecHitsSorted",
      "desc": "No documentation"
     },
     {
      "instance": "reducedEcalRecHitsEB",
      "container": "EcalRecHitsSorted",
      "desc": "No documentation"
     },
     {
      "instance": "correctedHybridSuperClusters",
      "container": "recoSuperClusters",
      "desc": "No documentation"
     },
     {
      "instance": "hybridSuperClusters",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "multi5x5SuperClusters",
      "container": "recoSuperClusters",
      "desc": "No documentation"
     },
     {
      "instance": "multi5x5SuperClusters",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "correctedMulti5x5SuperClustersWithPreshower",
      "container": "recoSuperClusters",
      "desc": "No documentation"
     },
     {
      "instance": "multi5x5SuperClustersWithPreshower",
      "container": "recoSuperClusters",
      "desc": "No documentation"
     }
    ]
  }
}
