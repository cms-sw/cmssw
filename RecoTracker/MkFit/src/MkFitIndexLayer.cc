#include "RecoTracker/MkFit/interface/MkFitIndexLayer.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>

void MkFitIndexLayer::insert(edm::ProductID id, size_t clusterIndex, int hit, int layer, const TrackingRecHit* hitPtr) {
  // mapping CMSSW->mkfit
  auto found = std::find_if(colls_.begin(), colls_.end(), [&](const auto& item) { return item.productID == id; });
  if (found == colls_.end()) {
    found = colls_.emplace(colls_.end(), id);
  }
  if (found->infos.size() <= clusterIndex) {
    found->infos.resize(clusterIndex + 1);
  }
  found->infos[clusterIndex] = HitInfo(hit, layer);

  // mapping mkfit->CMSSW
  if (layer >= static_cast<int>(hits_.size())) {
    hits_.resize(layer + 1);
  }
  if (hit >= static_cast<int>(hits_[layer].size())) {
    hits_[layer].resize(hit + 1);
  }
  hits_[layer][hit].ptr = hitPtr;
  hits_[layer][hit].clusterIndex = clusterIndex;
}

const MkFitIndexLayer::HitInfo& MkFitIndexLayer::get(edm::ProductID id, size_t clusterIndex) const {
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
