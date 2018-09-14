#ifndef RecoECAL_ECALClusters_ClusterEtLess_h
#define RecoECAL_ECALClusters_ClusterEtLess_h

namespace reco {
    class CaloCluster;
}

// Less than operator for sorting EcalRecHits according to energy.
bool isClusterEtLess(const reco::CaloCluster& x, const reco::CaloCluster& y);

#endif

