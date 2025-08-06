#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

bool SiStripMatchedRecHit2D::sharesInput(const TrackingRecHit* other, SharedInputType what) const {
  if (what == all && (geographicalId() != other->geographicalId()))
    return false;

  if (!sameDetModule(*other))
    return false;

  if (trackerHitRTTI::isMatched(*other)) {
    const SiStripMatchedRecHit2D* otherMatched = static_cast<const SiStripMatchedRecHit2D*>(other);
    return sharesClusters(*this, *otherMatched, what);
  }

  if (what == all)
    return false;
  // what about multi ???
  if (!trackerHitRTTI::isFromDet(*other))
    return false;

  auto const& otherClus = reinterpret_cast<const BaseTrackerRecHit*>(other)->firstClusterRef();
  if (monoClusterRef().id() == otherClus.id() || stereoClusterRef().id() == otherClus.id())
    return (otherClus == stereoClusterRef()) || (otherClus == monoClusterRef());
  else {
    bool stereoOverlap = otherClus.stripOverlap(stereoClusterRef());
    bool monoOverlap = otherClus.stripOverlap(monoClusterRef());
    return (stereoOverlap || monoOverlap);
  }
}

bool SiStripMatchedRecHit2D::sharesInput(TrackerSingleRecHit const& other) const {
  auto const& otherClus = other.firstClusterRef();
  if (monoClusterRef().id() == otherClus.id() || stereoClusterRef().id() == otherClus.id())
    return (otherClus == stereoClusterRef()) || (otherClus == monoClusterRef());
  else {
    const bool sameDetId = sameDetModule(other);
    bool stereoOverlap = (sameDetId) ? otherClus.stripOverlap(stereoClusterRef()) : false;
    bool monoOverlap = (sameDetId) ? otherClus.stripOverlap(monoClusterRef()) : false;
    return (stereoOverlap || monoOverlap);
  }
}

// it does not have components anymore...
std::vector<const TrackingRecHit*> SiStripMatchedRecHit2D::recHits() const {
  std::vector<const TrackingRecHit*> rechits;
  return rechits;
}

std::vector<TrackingRecHit*> SiStripMatchedRecHit2D::recHits() {
  std::vector<TrackingRecHit*> rechits;
  return rechits;
}
