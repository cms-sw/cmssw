#ifndef RecoTracker_MkFit_MkFitHitWrapper_h
#define RecoTracker_MkFit_MkFitHitWrapper_h

#include "DataFormats/Provenance/interface/ProductID.h"

#include <memory>
#include <vector>

namespace mkfit {
  class Hit;
  using HitVec = std::vector<Hit>;
}  // namespace mkfit

class MkFitHitWrapper {
public:
  MkFitHitWrapper();
  ~MkFitHitWrapper();

  MkFitHitWrapper(MkFitHitWrapper const&) = delete;
  MkFitHitWrapper& operator=(MkFitHitWrapper const&) = delete;
  MkFitHitWrapper(MkFitHitWrapper&&);
  MkFitHitWrapper& operator=(MkFitHitWrapper&&);

  void setClustersID(edm::ProductID id) { clustersID_ = id; }
  edm::ProductID clustersID() const { return clustersID_; }

  mkfit::HitVec& hits() { return hits_; }
  mkfit::HitVec const& hits() const { return hits_; }

private:
  // Vector is indexed by the cluster index
  mkfit::HitVec hits_;
  edm::ProductID clustersID_;
};

#endif
