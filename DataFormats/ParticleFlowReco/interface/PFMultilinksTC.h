#ifndef __PFMultilinksTC__
#define __PFMultilinksTC__

// Done by Glowinski & Gouzevitch

#include <vector>
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

namespace reco {

  /// \brief Abstract This class is used by the KDTree Track / Ecal Cluster
  /// linker to store all found links.
  ///
  struct PFMultilink {
    PFMultilink(const reco::PFClusterRef& clusterref) : trackRef(), clusterRef(clusterref) {}
    PFMultilink(const reco::PFRecTrackRef& trackref) : trackRef(trackref), clusterRef() {}
    reco::PFRecTrackRef trackRef;
    reco::PFClusterRef clusterRef;
  };
  /// collection of PFSuperCluster objects
  typedef std::vector<PFMultilink> PFMultilinksType;
  class PFMultiLinksTC {
  public:
    bool isValid;
    PFMultilinksType linkedPFObjects;

  public:
    PFMultiLinksTC(bool isvalid = false) : isValid(isvalid) {}
  };
}  // namespace reco

#endif
