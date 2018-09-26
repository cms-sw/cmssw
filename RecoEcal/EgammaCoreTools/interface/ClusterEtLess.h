#ifndef RecoECAL_ECALClusters_ClusterEtLess_h
#define RecoECAL_ECALClusters_ClusterEtLess_h

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

// Less than operator for sorting EcalRecHits according to energy.
inline bool isClusterEtLess(const reco::CaloCluster& x, const reco::CaloCluster& y)
{
  return ( (x.energy() * sin(x.position().theta())) < (y.energy() * sin(y.position().theta())) ) ;
}

#endif

