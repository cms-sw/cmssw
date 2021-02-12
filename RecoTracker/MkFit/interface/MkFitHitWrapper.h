#ifndef RecoTracker_MkFit_MkFitHitWrapper_h
#define RecoTracker_MkFit_MkFitHitWrapper_h

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

  mkfit::HitVec& pixelHits() { return *pixelHits_; }
  mkfit::HitVec const& pixelHits() const { return *pixelHits_; }

  mkfit::HitVec& outerHits() { return *outerHits_; }
  mkfit::HitVec const& outerHits() const { return *outerHits_; }

private:
  std::unique_ptr<mkfit::EventOfHits> eventOfHits_;

  // using unique_ptr to guarantee the address of the HitVec doesn't change in moves
  // EvenfOfHits relies on that
  // Vectors are indexed by the cluster index
  std::unique_ptr<mkfit::HitVec> pixelHits_;
  std::unique_ptr<mkfit::HitVec> outerHits_;
};

#endif
