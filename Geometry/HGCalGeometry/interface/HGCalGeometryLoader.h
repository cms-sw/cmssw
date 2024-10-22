#ifndef GeometryHGCalGeometryHGCalGeometryLoader_h
#define GeometryHGCalGeometryHGCalGeometryLoader_h
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/FlatHexagon.h"

class HGCalTopology;
class HGCalGeometry;

class HGCalGeometryLoader {
public:
  typedef CaloCellGeometry::CCGFloat CCGFloat;
  typedef std::vector<float> ParmVec;

  HGCalGeometryLoader();
  ~HGCalGeometryLoader() = default;

  HGCalGeometry* build(const HGCalTopology&);

private:
  void buildGeom(const ParmVec&, const HepGeom::Transform3D&, const DetId&, HGCalGeometry*, int mode);

  const double twoBysqrt3_;
  int parametersPerShape_;
};

#endif
