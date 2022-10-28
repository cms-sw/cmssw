#ifndef RecoTrackerDeDx_GenericTruncatedAverageDeDxEstimator_h
#define RecoTrackerDeDx_GenericTruncatedAverageDeDxEstimator_h

#include "RecoTracker/DeDx/interface/BaseDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include <numeric>

class GenericTruncatedAverageDeDxEstimator : public BaseDeDxEstimator {
public:
  GenericTruncatedAverageDeDxEstimator(const edm::ParameterSet& iConfig) {
    fraction_ = iConfig.getParameter<double>("fraction");
    expo_ = iConfig.getParameter<double>("exponent");
    truncate_ = iConfig.getParameter<bool>("truncate");
  }

  std::pair<float, float> dedx(const reco::DeDxHitCollection& Hits) override {
    int first = 0, last = Hits.size();
    if (truncate_) {
      if (fraction_ > 0) {  // truncate high charge ones
        last -= int(Hits.size() * fraction_);
      } else if (fraction_ < 0) {
        first += int(Hits.size() * (-fraction_));
      }
    }
    double sumdedx = 0;
    for (int i = first; i < last; i++) {
      sumdedx += pow(Hits[i].charge(), expo_);
    }
    double avrdedx = (last - first) ? pow(sumdedx / (last - first), 1.0 / expo_) : 0.0;
    return std::make_pair(avrdedx, -1);
  }

private:
  float fraction_, expo_;
  bool truncate_;
};

#endif
