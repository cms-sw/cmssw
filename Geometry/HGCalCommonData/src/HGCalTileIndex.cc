#include "Geometry/HGCalCommonData/interface/HGCalTileIndex.h"

int32_t HGCalTileIndex::tileIndex(int32_t layer, int32_t ring, int32_t phi) {
  int32_t id(0);
  id |= (((phi & kHGCalPhiMask) << kHGCalPhiOffset) | ((ring & kHGCalRingMask) << kHGCalRingOffset) |
         ((layer & kHGCalLayerMask) << kHGCalLayerOffset));
  return id;
}

int32_t HGCalTileIndex::tileLayer(int32_t id) { return ((id >> kHGCalLayerOffset) & kHGCalLayerMask); }

int32_t HGCalTileIndex::tileRing(int32_t id) { return ((id >> kHGCalRingOffset) & kHGCalRingMask); }

int32_t HGCalTileIndex::tilePhi(int32_t id) { return ((id >> kHGCalPhiOffset) & kHGCalPhiMask); }
