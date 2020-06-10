#include "Geometry/HGCalCommonData/interface/HGCalTileIndex.h"

const int32_t kHGCalLayerOffset = 18;
const int32_t kHGCalLayerMask = 0x1F;
const int32_t kHGCalPhiOffset = 0;
const int32_t kHGCalPhiMask = 0x1FF;
const int32_t kHGCalRingOffset = 9;
const int32_t kHGCalRingMask = 0x1FF;

int32_t HGCalTileIndex::tileIndex(const int32_t& layer, const int32_t& ring, const int32_t& phi) {
  int32_t id(0);
  id |= (((phi & kHGCalPhiMask) << kHGCalPhiOffset) |
	 ((ring & kHGCalRingMask) << kHGCalRingOffset) |
	 ((layer & kHGCalLayerMask) << kHGCalLayerOffset));
  return id;
}

int32_t HGCalTileIndex::tileLayer(const int32_t& id) { return ((id >> kHGCalLayerOffset) & kHGCalLayerMask); }

int32_t HGCalTileIndex::tileRing(const int32_t& id) { return ((id >> kHGCalRingOffset) & kHGCalRingMask);
}

int32_t HGCalTileIndex::tilePhi(const int32_t& id) { return ((id >> kHGCalPhiOffset) & kHGCalPhiMask);
}
