#ifndef OrderedHitTriplet_H
#define OrderedHitTriplet_H 


/** \class OrderedHitTriplet 
 * Associate 3 RecHits into hit triplet of InnerHit,MiddleHit,OuterHit
 */

#include "RecoTracker/TkHitPairs/interface/OrderedHitPair.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHit.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"

class OrderedHitTriplet : public SeedingHitSet {

public:
  typedef OrderedHitPair::InnerHit InnerHit;
  typedef OrderedHitPair::OuterHit OuterHit;
  typedef ctfseeding::SeedingHit MiddleHit;

  OrderedHitTriplet( const InnerHit & ih, const MiddleHit & mh, const OuterHit & oh) {
    theHits.push_back(ih);
    theHits.push_back(mh);
    theHits.push_back(oh);
  } 

  const InnerHit  &  inner() const { return theHits[0]; }
  const MiddleHit & middle() const { return theHits[1]; }
  const OuterHit  &  outer() const { return theHits[2]; }

};

#endif
