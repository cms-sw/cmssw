#ifndef OrderedHitTriplets_H
#define OrderedHitTriplets_H

#include <vector>
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplet.h"
#include "RecoTracker/TkSeedingLayers/interface/OrderedSeedingHits.h"
#include <vector>

class OrderedHitTriplets : public std::vector<OrderedHitTriplet>, public OrderedSeedingHits {
public:

  virtual ~OrderedHitTriplets(){}

  virtual unsigned int size() const { return std::vector<OrderedHitTriplet>::size(); }

  virtual const OrderedHitTriplet & operator[](unsigned int i) const {
    return std::vector<OrderedHitTriplet>::operator[](i);
  }

};
#endif
