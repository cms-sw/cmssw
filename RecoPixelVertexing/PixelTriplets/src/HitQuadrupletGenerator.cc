#include "RecoPixelVertexing/PixelTriplets/interface/HitQuadrupletGenerator.h"

HitQuadrupletGenerator::HitQuadrupletGenerator(unsigned int nSize): localRA(nSize) {}

const OrderedHitSeeds & HitQuadrupletGenerator::run(
    const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es)
{
  assert(theQuadruplets.empty());assert(theQuadruplets.capacity()==0);
  theQuadruplets.reserve(localRA.upper());
  hitQuadruplets(region, theQuadruplets, ev, es);
  localRA.update(theQuadruplets.size());
  theQuadruplets.shrink_to_fit();
  return theQuadruplets;
}

void HitQuadrupletGenerator::clear()
{
  theQuadruplets.clear(); theQuadruplets.shrink_to_fit();
}

