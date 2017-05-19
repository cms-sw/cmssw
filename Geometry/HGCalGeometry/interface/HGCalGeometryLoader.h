#ifndef GeometryHGCalGeometryHGCalGeometryLoader_h
#define GeometryHGCalGeometryHGCalGeometryLoader_h
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/FlatTrd.h"

class HGCalTopology;
class HGCalGeometry;

class HGCalGeometryLoader {

public:
  typedef CaloCellGeometry::CCGFloat CCGFloat;
  typedef std::vector<float> ParmVec;

  HGCalGeometryLoader ();
  ~HGCalGeometryLoader ();

  HGCalGeometry* build(const HGCalTopology& );

private:
  void buildGeom(const ParmVec&, const HepGeom::Transform3D&, const DetId&,
		 HGCalGeometry*);

};

#endif
