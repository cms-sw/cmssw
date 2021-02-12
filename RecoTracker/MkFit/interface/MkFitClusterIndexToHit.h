#ifndef RecoTracker_MkFit_MkFitClusterIndexToHit_h
#define RecoTracker_MkFit_MkFitClusterIndexToHit_h

#include <vector>

class TrackingRecHit;

class MkFitClusterIndexToHit {
public:
  MkFitClusterIndexToHit() = default;

  std::vector<TrackingRecHit const *> &pixelHits() { return pixelHits_; }
  std::vector<TrackingRecHit const *> const &pixelHits() const { return pixelHits_; }

  std::vector<TrackingRecHit const *> &outerHits() { return outerHits_; }
  std::vector<TrackingRecHit const *> const &outerHits() const { return outerHits_; }

private:
  // Indexed by cluster index
  std::vector<TrackingRecHit const *> pixelHits_;
  std::vector<TrackingRecHit const *> outerHits_;
};

#endif
