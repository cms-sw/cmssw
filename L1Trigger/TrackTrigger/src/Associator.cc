#include "L1Trigger/TrackTrigger/interface/Associator.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include <map>
#include <vector>
#include <deque>
#include <utility>
#include <numeric>
#include <algorithm>

namespace tt {

  // returns primary TP
  TPPtr Associator::getPrimaryTP(const TPPtr& tpPtr) const {
    const TVRef& tvRefParent = tpPtr->parentVertex();
    if (tvRefParent->nSourceTracks() > 0) {
      const TPRef& tpRefParent = *tvRefParent->sourceTracks_begin();
      const TPPtr tpPtrParent = edm::refToPtr(tpRefParent);
      return this->getPrimaryTP(tpPtrParent);
    }
    return tpPtr;
  }

  // checks if stub collection is considered forming a reconstructable track
  bool Associator::reconstructable(const std::vector<TTStubRef>& ttStubRefs) const {
    std::set<int> hitPattern;
    std::set<int> hitPatternPS;
    for (const TTStubRef& ttStubRef : ttStubRefs) {
      const int layerId = setup_->layerId(ttStubRef);
      hitPattern.insert(layerId);
      if (setup_->psModule(ttStubRef))
        hitPatternPS.insert(layerId);
    }
    if (static_cast<int>(hitPattern.size()) < config_.minLayers_)
      return false;
    if (static_cast<int>(hitPatternPS.size()) < config_.minLayersPS_)
      return false;
    return true;
  }

  // Get all TPs that are matched to these stubs in at least 'tpMinLayers' layers and 'tpMinLayersPS' ps layers
  std::vector<TPPtr> Associator::associate(const std::vector<TTStubRef>& ttStubRefs) const {
    // checks if min matching criteria are met
    auto valid = [this](const std::pair<std::set<int>, std::set<int>>& p) {
      if (static_cast<int>(p.first.size()) < config_.minLayersGood_)
        return false;
      if (static_cast<int>(p.second.size()) < config_.minLayersGoodPS_)
        return false;
      return true;
    };
    // count associated layer for each TP
    std::map<TPPtr, std::pair<std::set<int>, std::set<int>>> m;
    for (const TTStubRef& ttStubRef : ttStubRefs) {
      for (const TPPtr& tpPtr : sa_->findTrackingParticlePtrs(ttStubRef)) {
        const int layerId = setup_->layerId(ttStubRef);
        m[tpPtr].first.insert(layerId);
        if (setup_->psModule(ttStubRef))
          m[tpPtr].second.insert(layerId);
      }
    }
    // count matched TPs
    auto acc = [valid](int sum, const auto& p) { return sum + valid(p.second) ? 1 : 0; };
    const int nTPs = std::accumulate(m.begin(), m.end(), 0, acc);
    std::vector<TPPtr> tpPtrs;
    tpPtrs.reserve(nTPs);
    // fill and return matched TPs
    for (const auto& p : m) {
      if (valid(p.second))
        tpPtrs.push_back(p.first);
    }
    return tpPtrs;
  }

  // Get all TPs that are matched to these stubs in at least 'tpMinLayers' layers and 'tpMinLayersPS' ps layers with not more then 'tpMaxBadStubs2S' not associated 2S stubs and not more then 'tpMaxBadStubsPS' associated PS stubs
  std::vector<TPPtr> Associator::associateFinal(const std::vector<TTStubRef>& ttStubRefs) const {
    // checks if max matching criteria are met
    auto inValid = [this, &ttStubRefs](TPPtr& tpPtr) {
      int bad2S(0);
      int badPS(0);
      for (const TTStubRef& ttStubRef : ttStubRefs) {
        const std::vector<TPPtr>& tpPtrs = sa_->findTrackingParticlePtrs(ttStubRef);
        if (std::find(tpPtrs.begin(), tpPtrs.end(), tpPtr) == tpPtrs.end())
          setup_->psModule(ttStubRef) ? badPS++ : bad2S++;
      }
      return (badPS > config_.maxLayersBadPS_ || badPS + bad2S > config_.maxLayersBad_);
    };
    // Get all TPs that are matched to these stubs in at least 'tpMinLayers' layers and 'tpMinLayersPS' ps layers
    std::vector<TPPtr> tpPtrs = this->associate(ttStubRefs);
    // remove TPs with more then 'tpMaxBadStubs2S' not associated 2S stubs and more then 'tpMaxBadStubsPS' not associated PS stubs
    tpPtrs.erase(std::remove_if(tpPtrs.begin(), tpPtrs.end(), inValid), tpPtrs.end());
    return tpPtrs;
  }

}  // namespace tt
