#ifndef RecoECAL_ECALClusters_ClusterEtLess_h
#define RecoECAL_ECALClusters_ClusterEtLess_h


#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

// Less than operator for sorting EcalRecHits according to energy.
class ClusterEtLess : public std::binary_function<reco::CaloCluster, reco::CaloCluster, bool>
{
 public:
  bool operator()(reco::CaloCluster x, reco::CaloCluster y)
    {
      return ( (x.energy() * sin(x.position().theta())) < (y.energy() * sin(y.position().theta())) ) ;
    }
};

#endif

