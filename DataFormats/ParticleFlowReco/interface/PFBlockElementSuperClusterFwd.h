#ifndef PFBlockElementSuperClusterFwd_H
#define PFBlockElementSuperClusterFwd_H
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
namespace reco {
  class PFBlockElementSuperCluster;
  typedef std::vector<reco::PFBlockElementSuperCluster> PFBlockElementSuperClusterCollection;
  typedef edm::Ref<reco::PFBlockElementSuperClusterCollection> PFBlockElementSuperClusterRef;
}  // namespace reco

#endif
