#ifndef Geometry_HGCalCommonData_HGCalTileIndex_h
#define Geometry_HGCalCommonData_HGCalTileIndex_h

#include <cmath>
#include <cstdint>

namespace HGCalTileIndex {
  int32_t tileIndex(int32_t layer, int32_t ring, int32_t phi);
  int32_t tileLayer(int32_t index);
  int32_t tileRing(int32_t index);
  int32_t tilePhi(int32_t index);
};  // namespace HGCalTileIndex

#endif
