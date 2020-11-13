#ifndef RecoTracker_MkFit_MkFitHitWrapper_h
#define RecoTracker_MkFit_MkFitHitWrapper_h

#include "RecoTracker/MkFit/interface/MkFitHitIndexMap.h"

#include <vector>

namespace mkfit {
  class Hit;
  class LayerNumberConverter;
  using HitVec = std::vector<Hit>;
}  // namespace mkfit

class MkFitHitWrapper {
public:
  MkFitHitWrapper();
  MkFitHitWrapper(MkFitHitIndexMap hitIndexMap, std::vector<mkfit::HitVec> hits);
  ~MkFitHitWrapper();

  MkFitHitWrapper(MkFitHitWrapper const&) = delete;
  MkFitHitWrapper& operator=(MkFitHitWrapper const&) = delete;
  MkFitHitWrapper(MkFitHitWrapper&&);
  MkFitHitWrapper& operator=(MkFitHitWrapper&&);

  MkFitHitIndexMap const& hitIndexMap() const { return hitIndexMap_; }
  std::vector<mkfit::HitVec> const& hits() const { return hits_; }

private:
  MkFitHitIndexMap hitIndexMap_;
  std::vector<mkfit::HitVec> hits_;
};

#endif
