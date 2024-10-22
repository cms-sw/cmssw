#ifndef Geometry_HGCalCommonData_HGCalWaferIndex_h
#define Geometry_HGCalCommonData_HGCalWaferIndex_h

#include <cmath>
#include <cstdint>

namespace HGCalWaferIndex {
  // Packs layer, u, v indices into a single word (useful for xml definition)
  int32_t waferIndex(int32_t layer, int32_t waferU, int32_t waferV, bool old = false);
  // Unpacks the layer number from the packed index
  int32_t waferLayer(const int32_t index);
  // Unpacks wafer U from the packed index
  int32_t waferU(const int32_t index);
  // Unpacks wafer V from the packed index
  int32_t waferV(const int32_t index);
  // Gets the used part of the index (Layer:u:v)
  int32_t waferCopy(const int32_t index);
  // Finds the index format (old:false or new:true)
  bool waferFormat(const int32_t index);
};  // namespace HGCalWaferIndex

#endif
