#ifndef Geometry_HGCalCommonData_HGCalGeomRotation_h
#define Geometry_HGCalCommonData_HGCalGeomRotation_h

class HGCalGeomRotation {
public:
  enum class SectorType { Sector120Degrees, Sector60Degrees };
  enum class WaferCentring { WaferCentred, CornerCentredY, CornerCentredMercedes };

  HGCalGeomRotation(SectorType sectorType) { sectorType_ = sectorType; };
  ~HGCalGeomRotation() = default;

  void uvMappingFromSector0(WaferCentring waferCentring, int& moduleU, int& moduleV, unsigned sector) const;
  unsigned uvMappingToSector0(WaferCentring waferCentring, int& moduleU, int& moduleV) const;

private:
  void uvMappingFrom120DegreeSector0(WaferCentring waferCentring, int& moduleU, int& moduleV, unsigned sector) const;
  void uvMappingFrom60DegreeSector0(WaferCentring waferCentring, int& moduleU, int& moduleV, unsigned sector) const;

  unsigned uvMappingTo120DegreeSector0(WaferCentring waferCentring, int& moduleU, int& moduleV) const;
  unsigned uvMappingTo60DegreeSector0(WaferCentring waferCentring, int& moduleU, int& moduleV) const;

  void RotateModule60DegreesAnticlockwise(int& moduleU, int& moduleV) const;
  void RotateModule60DegreesClockwise(int& moduleU, int& moduleV) const;
  void RotateModule120DegreesAnticlockwise(int& moduleU, int& moduleV, int offset) const;
  void RotateModule120DegreesClockwise(int& moduleU, int& moduleV, int offset) const;

  SectorType sectorType_;
};

#endif
