#ifndef RecoTracker_LST_interface_LSTPhase2OTHitsInput_h
#define RecoTracker_LST_interface_LSTPhase2OTHitsInput_h

#include <memory>
#include <vector>

#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"

class LSTPhase2OTHitsInput {
public:
  LSTPhase2OTHitsInput() = default;
  LSTPhase2OTHitsInput(std::vector<unsigned int> const detId,
                       std::vector<float> const x,
                       std::vector<float> const y,
                       std::vector<float> const z,
                       std::vector<TrackingRecHit const*> const hits)
      : detId_(std::move(detId)), x_(std::move(x)), y_(std::move(y)), z_(std::move(z)), hits_(std::move(hits)) {}

  std::vector<unsigned int> const& detId() const { return detId_; }
  std::vector<float> const& x() const { return x_; }
  std::vector<float> const& y() const { return y_; }
  std::vector<float> const& z() const { return z_; }
  std::vector<TrackingRecHit const*> const& hits() const { return hits_; }

private:
  std::vector<unsigned int> detId_;
  std::vector<float> x_;
  std::vector<float> y_;
  std::vector<float> z_;
  std::vector<TrackingRecHit const*> hits_;
};

#endif
