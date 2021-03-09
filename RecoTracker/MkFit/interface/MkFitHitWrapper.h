#ifndef RecoTracker_MkFit_MkFitHitWrapper_h
#define RecoTracker_MkFit_MkFitHitWrapper_h

#include "DataFormats/Provenance/interface/ProductID.h"

#include <memory>
#include <vector>

namespace mkfit {
  class EventOfHits;
  class TrackerInfo;
  class Hit;
  using HitVec = std::vector<Hit>;
}  // namespace mkfit

class MkFitHitWrapper {
public:
  MkFitHitWrapper();
  MkFitHitWrapper(mkfit::TrackerInfo const& trackerInfo);
  ~MkFitHitWrapper();

  MkFitHitWrapper(MkFitHitWrapper const&) = delete;
  MkFitHitWrapper& operator=(MkFitHitWrapper const&) = delete;
  MkFitHitWrapper(MkFitHitWrapper&&);
  MkFitHitWrapper& operator=(MkFitHitWrapper&&);

  mkfit::EventOfHits& eventOfHits() { return *eventOfHits_; }
  mkfit::EventOfHits const& eventOfHits() const { return *eventOfHits_; }

  void setPixelClustersID(edm::ProductID id) { pixelClustersID_ = id; }
  edm::ProductID pixelClustersID() const { return pixelClustersID_; }

  void setOuterClustersID(edm::ProductID id) { outerClustersID_ = id; }
  edm::ProductID outerClustersID() const { return outerClustersID_; }

  mkfit::HitVec& pixelHits() { return *pixelHits_; }
  mkfit::HitVec const& pixelHits() const { return *pixelHits_; }

  mkfit::HitVec& outerHits() { return *outerHits_; }
  mkfit::HitVec const& outerHits() const { return *outerHits_; }

  std::vector<float>& stripClusterCharge() { return stripClusterCharge_; }
  void stripClusterChargeCut(float minThreshold, std::vector<bool>& mask) const;

private:
  std::unique_ptr<mkfit::EventOfHits> eventOfHits_;

  // using unique_ptr to guarantee the address of the HitVec doesn't change in moves
  // EvenfOfHits relies on that
  // Vectors are indexed by the cluster index
  std::unique_ptr<mkfit::HitVec> pixelHits_;
  std::unique_ptr<mkfit::HitVec> outerHits_;

  std::vector<float> stripClusterCharge_;

  edm::ProductID pixelClustersID_;
  edm::ProductID outerClustersID_;
};

#endif
