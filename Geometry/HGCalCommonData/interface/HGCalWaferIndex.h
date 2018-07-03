#ifndef Geometry_HGCalCommonData_HGCalWaferIndex_h
#define Geometry_HGCalCommonData_HGCalWaferIndex_h

#include <cmath>
#include <cstdint>

class HGCalWaferIndex {

public:
  
  HGCalWaferIndex() {}
  ~HGCalWaferIndex() {}
  static int32_t waferIndex(int32_t layer, int32_t waferU, int32_t waferV);
  static int32_t waferLayer(const int32_t index);
  static int32_t waferU(const int32_t index);
  static int32_t waferV(const int32_t index);

};

#endif
