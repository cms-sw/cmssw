#ifndef HitTripletGenerator_H
#define HitTripletGenerator_H

/** abstract interface for generators of hit triplets pairs
 *  compatible with a TrackingRegion.
 */

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplets.h"

class TrackingRegion;
namespace edm { class Event; class EventSetup; }
#include <vector>

class HitTripletGenerator : public OrderedHitsGenerator {
public:

  HitTripletGenerator(unsigned int size=500);

  virtual ~HitTripletGenerator() { }

  virtual const OrderedHitTriplets & run(
    const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es);

  // temporary interface, for bckwd compatibility
  virtual void hitTriplets( const TrackingRegion& reg, OrderedHitTriplets & prs,
       const edm::EventSetup& es){}

  virtual void hitTriplets( const TrackingRegion& reg, OrderedHitTriplets & prs,
      const edm::Event & ev,  const edm::EventSetup& es) = 0;

  virtual void clear();

private:
  OrderedHitTriplets theTriplets;

};


#endif
