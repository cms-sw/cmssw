#ifndef OrderedHitTriplet_H
#define OrderedHitTriplet_H 


/** \class OrderedHitTriplet 
 * Associate 3 RecHits into hit triplet of InnerHit,MiddleHit,OuterHit
 */

#include "RecoTracker/TkHitPairs/interface/OrderedHitPair.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"

class OrderedHitTriplet : public SeedingHitSet {

public:
  typedef OrderedHitPair::InnerHit InnerHit;
  typedef OrderedHitPair::OuterHit OuterHit;
  typedef ctfseeding::SeedingHit MiddleHit;

  typedef OrderedHitPair::InnerRecHit InnerRecHit;
  typedef OrderedHitPair::OuterRecHit OuterRecHit;
  typedef TransientTrackingRecHit::ConstRecHitPointer MiddleRecHit;


  OrderedHitTriplet( const InnerHit & ih, const MiddleHit & mh, const OuterHit & oh) {
    add(ih); add(mh); add(oh);
  } 

  OrderedHitTriplet( const InnerRecHit & ih, const MiddleRecHit & mh, const OuterRecHit & oh) {
    add(ih); add(mh); add(oh);
  }

  const InnerRecHit  &  inner() const { return theRecHits[0]; }
  const MiddleRecHit & middle() const { return theRecHits[1]; }
  const OuterRecHit  &  outer() const { return theRecHits[2]; }

};

#endif
