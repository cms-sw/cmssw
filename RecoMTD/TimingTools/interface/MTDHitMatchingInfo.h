#ifndef RecoMTD_TimingTools_MTDHitMatchingInfo_h
#define RecoMTD_TimingTools_MTDHitMatchingInfo_h

#include <limits>

#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"

namespace mtd {

  class MTDHitMatchingInfo {
  public:
    MTDHitMatchingInfo() {
      hit = nullptr;
      estChi2 = std::numeric_limits<float>::max();
      timeChi2 = std::numeric_limits<float>::max();
    }

    //Operator used to sort the hits while performing the matching step at the MTD
    inline bool operator<(const MTDHitMatchingInfo& m2) const {
      //only for good matching in time use estChi2, otherwise use mostly time compatibility
      constexpr float chi2_cut = 10.f;
      constexpr float low_weight = 3.f;
      constexpr float high_weight = 8.f;
      if (timeChi2 < chi2_cut && m2.timeChi2 < chi2_cut)
        return chi2(low_weight) < m2.chi2(low_weight);
      else
        return chi2(high_weight) < m2.chi2(high_weight);
    }

    inline float chi2(float timeWeight = 1.f) const { return estChi2 + timeWeight * timeChi2; }

    const MTDTrackingRecHit* hit;
    float estChi2;
    float timeChi2;
  };

}  // namespace mtd

#endif
