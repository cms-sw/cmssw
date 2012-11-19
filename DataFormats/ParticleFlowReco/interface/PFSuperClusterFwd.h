#ifndef ParticleFlowReco_PFSuperClusterFwd_h
#define ParticleFlowReco_PFSuperClusterFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class PFSuperCluster;

  /// collection of PFSuperCluster objects
  typedef std::vector<PFSuperCluster> PFSuperClusterCollection;

  /// persistent reference to PFSuperCluster objects
  typedef edm::Ref<PFSuperClusterCollection> PFSuperClusterRef;

  /// reference to PFSuperCluster collection
  typedef edm::RefProd<PFSuperClusterCollection> PFSuperClusterRefProd;

  /// vector of references to PFSuperCluster objects all in the same collection
  typedef edm::RefVector<PFSuperClusterCollection> PFSuperClusterRefVector;

  /// iterator over a vector of references to PFSuperCluster objects
  typedef PFSuperClusterRefVector::iterator PFSuperCluster_iterator;
}

#endif
