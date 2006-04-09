#ifndef EgammaReco_PreShowerClusterFwd_h
#define EgammaReco_PreShowerClusterFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
// #include "DataFormats/Common/interface/ExtCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace reco {
  class PreShowerCluster;
  /// collection of PreShowerCluster objects
  typedef std::vector<PreShowerCluster> PreShowerClusterCollection;
  /// persistent reference to PreShowerCluster objects
  typedef edm::Ref<PreShowerClusterCollection> PreShowerClusterRef;
  /// reference to PreShowerCluster collection
  typedef edm::RefProd<PreShowerClusterCollection> PreShowerClusterRefProd;
  /// vector of references to PreShowerCluster objects all in the same collection
  typedef edm::RefVector<PreShowerClusterCollection> PreShowerClusterRefVector;
  /// iterator over a vector of references to PreShowerCluster objects
  typedef PreShowerClusterRefVector::iterator PreShowerCluster_iterator;
}

#endif
