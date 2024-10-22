#ifndef GeometryHGCalGeometryHGCalTBGeometryLoader_h
#define GeometryHGCalGeometryHGCalTBGeometryLoader_h
#include "Geometry/HGCalGeometry/interface/HGCalTBGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/FlatHexagon.h"

class HGCalTBTopology;
class HGCalTBGeometry;

class HGCalTBGeometryLoader {
public:
  typedef CaloCellGeometry::CCGFloat CCGFloat;
  typedef std::vector<float> ParmVec;

  HGCalTBGeometryLoader();
  ~HGCalTBGeometryLoader() = default;

  HGCalTBGeometry* build(const HGCalTBTopology&);

private:
  void buildGeom(const ParmVec&, const HepGeom::Transform3D&, const DetId&, HGCalTBGeometry*);

  const double twoBysqrt3_;
  int parametersPerShape_;
};

#endif
