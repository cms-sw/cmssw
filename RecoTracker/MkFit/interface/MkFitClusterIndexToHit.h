#ifndef RecoTracker_MkFit_MkFitClusterIndexToHit_h
#define RecoTracker_MkFit_MkFitClusterIndexToHit_h

#include <vector>

class TrackingRecHit;

class MkFitClusterIndexToHit {
public:
  MkFitClusterIndexToHit() = default;

  std::vector<TrackingRecHit const*>& hits() { return hits_; }
  std::vector<TrackingRecHit const*> const& hits() const { return hits_; }

private:
  // Indexed by cluster index
  std::vector<TrackingRecHit const*> hits_;
};

#endif
