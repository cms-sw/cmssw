#include "RecoEcal/EgammaCoreTools/interface/ClusterEtLess.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

// Less than operator for sorting EcalRecHits according to energy.
bool isClusterEtLess(const reco::CaloCluster& x, const reco::CaloCluster& y)
{
  return ( (x.energy() * sin(x.position().theta())) < (y.energy() * sin(y.position().theta())) ) ;
}
