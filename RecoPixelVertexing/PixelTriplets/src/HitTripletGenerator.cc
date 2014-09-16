#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGenerator.h"

HitTripletGenerator::HitTripletGenerator(unsigned int nSize) : localRA(nSize)
{
//  theTriplets.reserve(nSize);
}

const OrderedHitTriplets & HitTripletGenerator::run(
    const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es)
{
  OrderedHitTriplets tmp; tmp.reserve(localRA.upper()); tmp.swap(theTriplets);
  hitTriplets(region, theTriplets, ev, es);
  localRA.update(theTriplets.size());
  theTriplets.shrink_to_fit();
  return theTriplets;
}

void HitTripletGenerator::clear() 
{
    OrderedHitTriplets tmp; tmp.swap(theTriplets);
//  theTriplets.clear();
}

