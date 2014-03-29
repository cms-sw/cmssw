#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

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

namespace {
  inline
  void throwError() {
    throw cms::Exception("Global coordinates missing from this TrackingRecHit used");
  }
}

const GeomDetUnit * TrackingRecHit::detUnit() const
{
  return dynamic_cast<const GeomDetUnit*>(det());
}


GlobalPoint TrackingRecHit::globalPosition() const { throwError(); return GlobalPoint();}
GlobalError TrackingRecHit::globalPositionError() const { throwError(); return GlobalError();}

float TrackingRecHit::errorGlobalR() const{ throwError(); return 0;}
float TrackingRecHit::errorGlobalZ() const{ throwError(); return 0;}
float TrackingRecHit::errorGlobalRPhi() const{ throwError(); return 0;}
