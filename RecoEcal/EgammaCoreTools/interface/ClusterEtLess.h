#ifndef RecoECAL_ECALClusters_ClusterEtLess_h
#define RecoECAL_ECALClusters_ClusterEtLess_h

namespace reco {
    class CaloCluster;
}

// Less than operator for sorting EcalRecHits according to energy.
bool isClusterEtLess(reco::CaloCluster x, reco::CaloCluster y);

#endif

