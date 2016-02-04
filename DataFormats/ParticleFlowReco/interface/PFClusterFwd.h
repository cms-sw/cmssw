#ifndef ParticleFlowReco_PFClusterFwd_h
#define ParticleFlowReco_PFClusterFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class PFCluster;

  /// collection of PFCluster objects
  typedef std::vector<PFCluster> PFClusterCollection;

  /// persistent reference to PFCluster objects
  typedef edm::Ref<PFClusterCollection> PFClusterRef;

  /// reference to PFCluster collection
  typedef edm::RefProd<PFClusterCollection> PFClusterRefProd;

  /// vector of references to PFCluster objects all in the same collection
  typedef edm::RefVector<PFClusterCollection> PFClusterRefVector;

  /// iterator over a vector of references to PFCluster objects
  typedef PFClusterRefVector::iterator PFCluster_iterator;
}

#endif
