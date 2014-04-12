#ifndef OrderedHitTriplet_H
#define OrderedHitTriplet_H 


/** \class OrderedHitTriplet 
 * Associate 3 RecHits into hit triplet of InnerHit,MiddleHit,OuterHit
 */

#include "RecoTracker/TkHitPairs/interface/OrderedHitPair.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"

class OrderedHitTriplet : public SeedingHitSet {

public:

  typedef SeedingHitSet::ConstRecHitPointer InnerRecHit;
  typedef SeedingHitSet::ConstRecHitPointer OuterRecHit;
  typedef SeedingHitSet::ConstRecHitPointer MiddleRecHit;


  OrderedHitTriplet( const InnerRecHit & ih, const MiddleRecHit & mh, const OuterRecHit & oh) : SeedingHitSet(ih,mh,oh){}

  InnerRecHit    inner() const { return get(0); }
  MiddleRecHit  middle() const { return get(1); }
  OuterRecHit    outer() const { return get(2); }

};

#endif
