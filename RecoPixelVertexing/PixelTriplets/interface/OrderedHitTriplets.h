#ifndef OrderedHitTriplets_H
#define OrderedHitTriplets_H

#include <vector>
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplet.h"
#include "RecoTracker/TkSeedingLayers/interface/OrderedSeedingHits.h"
#include <vector>

class OrderedHitTriplets : public std::vector<OrderedHitTriplet>, public OrderedSeedingHits {
public:

  ~OrderedHitTriplets() override{}

  unsigned int size() const override { return std::vector<OrderedHitTriplet>::size(); }

  const OrderedHitTriplet & operator[](unsigned int i) const override {
    return std::vector<OrderedHitTriplet>::operator[](i);
  }

};
#endif
