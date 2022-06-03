#ifndef Geometry_HGCalCommonData_HGCalTileIndex_h
#define Geometry_HGCalCommonData_HGCalTileIndex_h

#include <cmath>
#include <cstdint>
#include <tuple>

namespace HGCalTileIndex {
  // Packs layer, ring, phi indices into a single word (useful for XML files)
  int32_t tileIndex(int32_t layer, int32_t ring, int32_t phi);
  // Unpacks Layer number from the packed index
  int32_t tileLayer(int32_t index);
  // Unpacks Ring number from the packed index
  int32_t tileRing(int32_t index);
  // Unpacks Phi number from the packed index
  int32_t tilePhi(int32_t index);
  // Packs tile type and SiPM size into a single word (useful for XML files)
  int32_t tileProperty(int32_t, int32_t);
  // Unpacks tile type from the packed word
  int32_t tileType(int32_t);
  // Unpacks SiPM size from the packed word
  int32_t tileSiPM(int32_t);
  // Gets cassette number from phi position
  int32_t tileCassette(int32_t, int32_t, int32_t, int32_t);
  // Packs 3 information for usage in xml file
  int32_t tilePack(int32_t ly, int32_t k1, int32_t k2);
  // Unpacks thos three information from the packed word
  std::tuple<int32_t, int32_t, int32_t> tileUnpack(int32_t index);
  // Sees if the tile exists or not depending the HEX information in flat file
  bool tileExist(const int32_t* hex, int32_t zside, int32_t phi);
};  // namespace HGCalTileIndex

#endif
