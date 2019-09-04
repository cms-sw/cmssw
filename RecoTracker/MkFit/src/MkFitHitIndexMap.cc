#include "RecoTracker/MkFit/interface/MkFitHitIndexMap.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>

namespace {
  template <typename T>
  auto resizeByClusterIndexImpl(T& cmsswToMkFit, edm::ProductID id, size_t clusterIndex) -> typename T::iterator {
    auto found = std::find_if(cmsswToMkFit.begin(), cmsswToMkFit.end(), [&](const auto& item) { return item.productID == id; });
    if (found == cmsswToMkFit.end()) {
      found = cmsswToMkFit.emplace(cmsswToMkFit.end(), id);
    }
    if (found->infos.size() <= clusterIndex) {
      found->infos.resize(clusterIndex + 1);
    }
    return found;
  }
}

void MkFitHitIndexMap::resizeByClusterIndex(edm::ProductID id, size_t clusterIndex) {
  resizeByClusterIndexImpl(colls_, id, clusterIndex);
}

void MkFitHitIndexMap::increaseLayerSize(int layer, size_t additionalSize) {
  if (layer >= static_cast<int>(hits_.size())) {
    hits_.resize(layer + 1);
  }
  hits_[layer].resize(hits_[layer].size() + additionalSize);
}

void MkFitHitIndexMap::insert(edm::ProductID id, size_t clusterIndex, int hit, int layer, const TrackingRecHit* hitPtr) {
  // mapping CMSSW->mkfit
  auto found = resizeByClusterIndexImpl(colls_, id, clusterIndex);
  found->infos[clusterIndex] = HitInfo(hit, layer);

  // mapping mkfit->CMSSW
  // when client calls increaseLayerSize() the two checks below are
  // redundant, but better to keep them
  if (layer >= static_cast<int>(hits_.size())) {
    hits_.resize(layer + 1);
  }
  if (hit >= static_cast<int>(hits_[layer].size())) {
    hits_[layer].resize(hit + 1);
  }
  hits_[layer][hit].ptr = hitPtr;
  hits_[layer][hit].clusterIndex = clusterIndex;
}

const MkFitHitIndexMap::HitInfo& MkFitHitIndexMap::get(edm::ProductID id, size_t clusterIndex) const {
  auto found = std::find_if(colls_.begin(), colls_.end(), [&](const auto& item) { return item.productID == id; });
  if (found == colls_.end()) {
    auto exp = cms::Exception("Assert");
    exp << "Encountered a seed with a hit having productID " << id
        << " which is not any of the input hit collections: ";
    for (const auto& elem : colls_) {
      exp << elem.productID << " ";
    }
    throw exp;
  }
  const HitInfo& ret = found->infos[clusterIndex];
  if (ret.index < 0) {
    throw cms::Exception("Assert") << "No hit index for cluster " << clusterIndex << " of collection " << id;
  }
  return ret;
}
