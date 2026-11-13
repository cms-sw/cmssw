#ifndef EgammaReco_SlimmedSuperClusterFwd_h
#define EgammaReco_SlimmedSuperClusterFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class SlimmedSuperCluster;

  /// collection of SuperCluser objectr
  typedef std::vector<SlimmedSuperCluster> SlimmedSuperClusterCollection;

  /// reference to an object in a collection of SlimmedSuperCluster objects
  typedef edm::Ref<SlimmedSuperClusterCollection> SlimmedSuperClusterRef;

  /// reference to a collection of SlimmedSuperCluster objects
  typedef edm::RefProd<SlimmedSuperClusterCollection> SlimmedSuperClusterRefProd;

  /// vector of references to objects in the same colletion of SlimmedSuperCluster objects
  typedef edm::RefVector<SlimmedSuperClusterCollection> SlimmedSuperClusterRefVector;

}  // namespace reco

#endif
