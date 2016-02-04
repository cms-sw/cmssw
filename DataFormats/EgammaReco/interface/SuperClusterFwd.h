#ifndef EgammaReco_SuperClusterFwd_h
#define EgammaReco_SuperClusterFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class SuperCluster;

  /// collection of SuperCluser objectr
  typedef std::vector<SuperCluster> SuperClusterCollection;

  /// reference to an object in a collection of SuperCluster objects
  typedef edm::Ref<SuperClusterCollection> SuperClusterRef;

  /// reference to a collection of SuperCluster objects
  typedef edm::RefProd<SuperClusterCollection> SuperClusterRefProd;

  /// vector of references to objects in the same colletion of SuperCluster objects
  typedef edm::RefVector<SuperClusterCollection> SuperClusterRefVector;

  /// iterator over a vector of reference to SuperClusters
  typedef SuperClusterRefVector::iterator superCluster_iterator;
}

#endif
