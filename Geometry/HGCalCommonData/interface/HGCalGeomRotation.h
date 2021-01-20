#ifndef Geometry_HGCalCommonData_HGCalGeomRotation_h
#define Geometry_HGCalCommonData_HGCalGeomRotation_h

class HGCalGeomRotation {
public:

  enum SectorType { Sector120Degrees, Sector60Degrees };
  enum WaferCentring { WaferCentred, CornerCentredY, CornerCentredMercedes };

  HGCalGeomRotation(SectorType sectorType){_sectorType = sectorType;};
  ~HGCalGeomRotation() {}

  void uvMappingFromSector0(WaferCentring waferCentring, int& moduleU, int& moduleV, unsigned sector) const;
  unsigned uvMappingToSector0(WaferCentring waferCentring, int& moduleU, int& moduleV) const;

private:

  void RotateModule60DegreesAnticlockwise(int& moduleU, int& moduleV) const;
  void RotateModule60DegreesClockwise(int& moduleU, int& moduleV) const;

  unsigned uvMappingTo120DegreeSector0(WaferCentring waferCentring, int& moduleU, int& moduleV) const;
  unsigned uvMappingTo60DegreeSector0(WaferCentring waferCentring, int& moduleU, int& moduleV) const;

  SectorType _sectorType;


};

#endif
