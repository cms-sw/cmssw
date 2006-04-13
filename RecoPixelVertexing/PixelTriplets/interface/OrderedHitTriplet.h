#ifndef OrderedHitTriplet_H
#define OrderedHitTriplet_H 


/** \class OrderedHitTriplet 
 * Associate 3 RecHits into hit triplet of InnerHit,MiddleHit,OuterHit
 */

#include <vector>

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoTracker/TkHitPairs/interface/OrderedHitPair.h"


class OrderedHitTriplet : public OrderedHitPair {

public:
  typedef TrackingRecHit MiddleHit;

/// constructor from InnerHit, MiddleHit and OuterHit
  explicit OrderedHitTriplet(
      const InnerHit * ih, const MiddleHit * mh,const OuterHit * oh) 
      : OrderedHitPair(ih,oh), theMiddleHit(mh) { }

  const MiddleHit * middle() const { return theMiddleHit; }

private:
  const MiddleHit * theMiddleHit;
};

#endif
