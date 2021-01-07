#ifndef Geometry_HGCalCommonData_HGCalTileIndex_h
#define Geometry_HGCalCommonData_HGCalTileIndex_h

#include <cmath>
#include <cstdint>

namespace HGCalTileIndex {
  int32_t tileIndex(int32_t layer, int32_t ring, int32_t phi);
  int32_t tileLayer(int32_t index);
  int32_t tileRing(int32_t index);
  int32_t tilePhi(int32_t index);
  int32_t tileProperty(int32_t, int32_t);
  int32_t tileType(int32_t);
  int32_t tileSiPM(int32_t);
  int32_t tilePack(int32_t k1, int32_t k2);
  std::pair<int32_t, int32_t> tileUnpack(int32_t index);
  bool tileExist(const int32_t* hex, int32_t zside, int32_t phi);
};  // namespace HGCalTileIndex

#endif
