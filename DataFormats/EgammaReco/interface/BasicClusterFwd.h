#ifndef EgammaReco_BasicClusterFwd_h
#define EgammaReco_BasicClusterFwd_h

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

#include <vector>

namespace reco {

  typedef CaloCluster BasicCluster;

  /// collection of BasicCluster objects
  typedef std::vector<BasicCluster> BasicClusterCollection;

  /// persistent reference to BasicCluster objects
  typedef edm::Ref<BasicClusterCollection> BasicClusterRef;

  /// reference to BasicCluster collection
  typedef edm::RefProd<BasicClusterCollection> BasicClusterRefProd;

  /// vector of references to BasicCluster objects all in the same collection
  typedef edm::RefVector<BasicClusterCollection> BasicClusterRefVector;

  /// iterator over a vector of references to BasicCluster objects
  typedef BasicClusterRefVector::iterator basicCluster_iterator;

}  // namespace reco

#endif
