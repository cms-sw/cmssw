#ifndef FastProjectedTrackerRecHit_H
#define FastProjectedTrackerRecHit_H

#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastSingleTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"

class FastProjectedTrackerRecHit : public FastTrackerRecHit {
public:
  FastProjectedTrackerRecHit(){};

  ~FastProjectedTrackerRecHit() override{};

  FastProjectedTrackerRecHit(const LocalPoint& pos,
                             const LocalError& err,
                             GeomDet const& idet,
                             FastSingleTrackerRecHit const& originalHit)
      : FastTrackerRecHit(pos,
                          err,
                          idet,
                          ProjectedSiStripRecHit2D::isMono(idet, *originalHit.det())
                              ? fastTrackerRecHitType::siStripProjectedMono2D
                              : fastTrackerRecHitType::siStripProjectedStereo2D),
        originalHit_(originalHit) {}

  const FastSingleTrackerRecHit& originalHit() const { return originalHit_; }
  FastProjectedTrackerRecHit* clone() const override {
    FastProjectedTrackerRecHit* p = new FastProjectedTrackerRecHit(*this);
    p->load();
    return p;
  }
  size_t nIds() const override { return 1; }
  int32_t id(size_t i = 0) const override { return originalHit().id(i); }
  int32_t eventId(size_t i = 0) const override { return originalHit().eventId(i); }
  size_t nSimTrackIds() const override {
    return originalHit_.nSimTrackIds();
  }  ///< see addSimTrackId(int32_t simTrackId)
  int32_t simTrackId(size_t i) const override {
    return originalHit_.simTrackId(i);
  }  ///< see addSimTrackId(int32_t simTrackId)
  int32_t simTrackEventId(size_t i) const override {
    return originalHit_.simTrackEventId(i);
    ;
  }  ///< see addSimTrackId(int32_t simTrackId)

  void setEventId(int32_t eventId) override { originalHit_.setEventId(eventId); }

  void setRecHitCombinationIndex(int32_t recHitCombinationIndex) override {
    FastTrackerRecHit::setRecHitCombinationIndex(recHitCombinationIndex);
    originalHit_.setRecHitCombinationIndex(recHitCombinationIndex);
  }

private:
  FastSingleTrackerRecHit originalHit_;
};

#endif
