#include "RecoTracker/MkFit/interface/MkFitHitIndexMap.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>

namespace {
  template <typename T>
  auto resizeByClusterIndexImpl(T& cmsswToMkFit, edm::ProductID id, size_t clusterIndex) -> typename T::iterator {
    auto found =
        std::find_if(cmsswToMkFit.begin(), cmsswToMkFit.end(), [&](const auto& item) { return item.productID == id; });
    if (found == cmsswToMkFit.end()) {
      found = cmsswToMkFit.emplace(cmsswToMkFit.end(), id);
    }
    if (found->mkFitHits.size() <= clusterIndex) {
      found->mkFitHits.resize(clusterIndex + 1);
    }
    return found;
  }
}  // namespace

void MkFitHitIndexMap::resizeByClusterIndex(edm::ProductID id, size_t clusterIndex) {
  resizeByClusterIndexImpl(cmsswToMkFit_, id, clusterIndex);
}

void MkFitHitIndexMap::increaseLayerSize(int layer, size_t additionalSize) {
  if (layer >= static_cast<int>(mkFitToCMSSW_.size())) {
    mkFitToCMSSW_.resize(layer + 1);
  }
  mkFitToCMSSW_[layer].resize(mkFitToCMSSW_[layer].size() + additionalSize);
}

void MkFitHitIndexMap::insert(edm::ProductID id, size_t clusterIndex, MkFitHit hit, const TrackingRecHit* hitPtr) {
  // mapping CMSSW->mkfit
  auto found = resizeByClusterIndexImpl(cmsswToMkFit_, id, clusterIndex);
  found->mkFitHits[clusterIndex] = hit;

  // mapping mkfit->CMSSW
  // when client calls increaseLayerSize() the two checks below are
  // redundant, but better to keep them
  if (hit.layer() >= static_cast<int>(mkFitToCMSSW_.size())) {
    mkFitToCMSSW_.resize(hit.layer() + 1);
  }
  auto& layer = mkFitToCMSSW_[hit.layer()];
  if (hit.index() >= static_cast<int>(layer.size())) {
    layer.resize(hit.index() + 1);
  }
  layer[hit.index()] = CMSSWHit(hitPtr, clusterIndex);
}

const MkFitHitIndexMap::MkFitHit& MkFitHitIndexMap::mkFitHit(edm::ProductID id, size_t clusterIndex) const {
  auto found =
      std::find_if(cmsswToMkFit_.begin(), cmsswToMkFit_.end(), [&](const auto& item) { return item.productID == id; });
  if (found == cmsswToMkFit_.end()) {
    auto exp = cms::Exception("Assert");
    exp << "Encountered a seed with a hit having productID " << id
        << " which is not any of the input hit collections: ";
    for (const auto& elem : cmsswToMkFit_) {
      exp << elem.productID << " ";
    }
    throw exp;
  }
  const MkFitHit& ret = found->mkFitHits.at(clusterIndex);
  if (ret.index() < 0) {
    throw cms::Exception("Assert") << "No hit index for cluster " << clusterIndex << " of collection " << id;
  }
  return ret;
}
