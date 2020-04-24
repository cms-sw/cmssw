#ifndef HitQuadrupletGenerator_H
#define HitQuadrupletGenerator_H

/** abstract interface for generators of hit triplets pairs
 *  compatible with a TrackingRegion.
 */

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitSeeds.h"

#include "FWCore/Utilities/interface/RunningAverage.h"

class TrackingRegion;
namespace edm { class Event; class EventSetup; }
#include <vector>

class HitQuadrupletGenerator : public OrderedHitsGenerator {
public:

  HitQuadrupletGenerator(unsigned int size=500);

  ~HitQuadrupletGenerator() override { }

  const OrderedHitSeeds & run(
    const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es) final;

  virtual void hitQuadruplets( const TrackingRegion& reg, OrderedHitSeeds& prs,
      const edm::Event & ev,  const edm::EventSetup& es) = 0;

  void clear() final;

private:
  OrderedHitSeeds theQuadruplets;
  edm::RunningAverage localRA;
};


#endif
