#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGenerator.h"

HitTripletGenerator::HitTripletGenerator(unsigned int nSize)
{
  theTriplets.reserve(nSize);
}

const OrderedHitTriplets & HitTripletGenerator::run(
    const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es)
{
  theTriplets.clear();
  hitTriplets(region, theTriplets, ev, es);
  return theTriplets;
}

void HitTripletGenerator::clear() 
{
  theTriplets.clear();
}

