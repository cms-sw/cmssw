#ifndef Geometry_HGCalCommonData_HGCalGeomRotation_h
#define Geometry_HGCalCommonData_HGCalGeomRotation_h


class HGCalGeomRotation {
public:
  HGCalGeomRotation();
  ~HGCalGeomRotation() {}

  enum WaferCentring { WaferCentred, CornerCentredY, CornerCentredMercedes };

  void uvMappingFromSector0(WaferCentring waferCentring, int& moduleU, int& moduleV, unsigned sector) const;
  unsigned uvMappingToSector0(WaferCentring waferCentring, int& moduleU, int& moduleV) const;

};

#endif
