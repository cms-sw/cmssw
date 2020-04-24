#ifndef GeometryHGCalGeometryFastTimeGeometryLoader_h
#define GeometryHGCalGeometryFastTimeGeometryLoader_h
#include "Geometry/HGCalGeometry/interface/FastTimeGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/FlatTrd.h"
#include "Geometry/CaloTopology/interface/FastTimeTopology.h"

class FastTimeGeometryLoader {

public:
  typedef CaloCellGeometry::CCGFloat CCGFloat;
  typedef std::vector<float> ParmVec;

  FastTimeGeometryLoader ();
  ~FastTimeGeometryLoader ();

  FastTimeGeometry* build(const FastTimeTopology& );

private:
  void buildGeom(const ParmVec&, const HepGeom::Transform3D&, const DetId&,
		 const FastTimeTopology&, FastTimeGeometry*);
};

#endif
