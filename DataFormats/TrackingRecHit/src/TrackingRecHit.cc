#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <string>
#include <typeinfo>

void
TrackingRecHit::recHitsV(std::vector<const TrackingRecHit*> & v) const {
  v = recHits();
}
void
TrackingRecHit::recHitsV(std::vector<TrackingRecHit*> & v) {
  v = recHits();
}


bool TrackingRecHit::sharesInput( const TrackingRecHit* other, SharedInputType what) const {
  //
  // for the time being: don't force implementation in all derived classes
  // but throw exception to indicate missing implementation
  //
  std::string msg("Missing implementation of TrackingRecHit::sharedInput in ");
  msg += typeid(*this).name();
  throw cms::Exception(msg);
  return false;
}

void TrackingRecHit::getKfComponents( KfComponentsHolder & holder ) const {
    holder.genericFill(*this);
}
