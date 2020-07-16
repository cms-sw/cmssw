#ifndef Geometry_HGCalCommonData_HGCalTileIndex_h
#define Geometry_HGCalCommonData_HGCalTileIndex_h

#include <cmath>
#include <cstdint>

class HGCalTileIndex {
public:
  HGCalTileIndex() {}
  ~HGCalTileIndex() {}
  static int32_t tileIndex(int32_t layer, int32_t ring, int32_t phi);
  static int32_t tileLayer(int32_t index);
  static int32_t tileRing(int32_t index);
  static int32_t tilePhi(int32_t index);

private:
  static constexpr int32_t kHGCalLayerOffset = 18;
  static constexpr int32_t kHGCalLayerMask = 0x1F;
  static constexpr int32_t kHGCalPhiOffset = 0;
  static constexpr int32_t kHGCalPhiMask = 0x1FF;
  static constexpr int32_t kHGCalRingOffset = 9;
  static constexpr int32_t kHGCalRingMask = 0x1FF;
};

#endif
