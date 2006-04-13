#ifndef HitTripletGenerator_H
#define HitTripletGenerator_H

/** abstract interface for generators of hit triplets pairs
 *  compatible with a TrackingRegion.
 */

#include <vector>
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplets.h"
#include "FWCore/Framework/interface/EventSetup.h"
class TrackingRegion;


class HitTripletGenerator {

public:
  virtual ~HitTripletGenerator() { }
  virtual OrderedHitTriplets hitTriplets( const TrackingRegion& region, 
      const edm::EventSetup& iSetup ) {
    OrderedHitTriplets triplets;
    hitTriplets(region, triplets, iSetup);
    return triplets;
  }
  virtual void hitTriplets(
      const TrackingRegion& region, OrderedHitTriplets & trs,
      const edm::EventSetup& iSetup) = 0;

};

#endif
