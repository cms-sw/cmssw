#ifndef Geometry_HGCalCommonData_HGCalProperty_h
#define Geometry_HGCalCommonData_HGCalProperty_h

#include <cmath>
#include <cstdint>

class HGCalProperty {
public:
  HGCalProperty() {}
  ~HGCalProperty() {}
  static int32_t waferProperty(const int32_t thick, const int32_t partial, const int32_t orient);
  static int32_t waferThick(const int32_t property);
  static int32_t waferPartial(const int32_t property);
  static int32_t waferOrient(const int32_t property);
  static int32_t tileProperty(const int32_t type, const int32_t sipm);
  static int32_t tileType(const int32_t property);
  static int32_t tileSiPM(const int32_t property);
};

#endif
