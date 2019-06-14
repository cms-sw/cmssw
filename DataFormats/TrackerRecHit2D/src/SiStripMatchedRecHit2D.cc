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
  return (otherClus == stereoClusterRef()) || (otherClus == monoClusterRef());
}

bool SiStripMatchedRecHit2D::sharesInput(TrackerSingleRecHit const& other) const {
  return other.sameCluster(monoClusterRef()) || other.sameCluster(stereoClusterRef());
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
