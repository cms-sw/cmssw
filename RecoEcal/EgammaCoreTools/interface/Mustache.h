#ifndef RecoEcal_EgammaCoreTools_Mustache_h
#define RecoEcal_EgammaCoreTools_Mustache_h

#include <vector>
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

namespace reco {

  class Mustache {
  public:
    void MustacheID(CaloClusterPtrVector& clusters, int & nclusters, float & EoutsideMustache);
    void MustacheID(const reco::SuperCluster& sc, int & nclusters, float & EoutsideMustache);
    
  };
  
}

#endif
