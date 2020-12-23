#include "Geometry/HGCalCommonData/interface/HGCalProperty.h"
#include "Geometry/HGCalCommonData/interface/HGCalTileIndex.h"

int32_t HGCalTileIndex::tileIndex(int32_t layer, int32_t ring, int32_t phi) {
  int32_t id(0);
  id |= (((phi & HGCalProperty::kHGCalPhiMask) << HGCalProperty::kHGCalPhiOffset) |
         ((ring & HGCalProperty::kHGCalRingMask) << HGCalProperty::kHGCalRingOffset) |
         ((layer & HGCalProperty::kHGCalLayerMask) << HGCalProperty::kHGCalLayerOffset));
  return id;
}

int32_t HGCalTileIndex::tileLayer(int32_t id) {
  return ((id >> HGCalProperty::kHGCalLayerOffset) & HGCalProperty::kHGCalLayerMask);
}

int32_t HGCalTileIndex::tileRing(int32_t id) {
  return ((id >> HGCalProperty::kHGCalRingOffset) & HGCalProperty::kHGCalRingMask);
}

int32_t HGCalTileIndex::tilePhi(int32_t id) {
  return ((id >> HGCalProperty::kHGCalPhiOffset) & HGCalProperty::kHGCalPhiMask);
}
