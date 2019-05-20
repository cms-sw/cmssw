#ifndef Geometry_GEMGeometry_ME0GeometryBuilderFromDDD_H
#define Geometry_GEMGeometry_ME0GeometryBuilderFromDDD_H

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include <string>
#include <map>
#include <vector>

class DDCompactView;
class DDFilteredView;
class ME0Geometry;
class ME0DetId;
class ME0Chamber;
class ME0Layer;
class ME0EtaPartition;
class MuonDDDConstants;

class ME0GeometryBuilderFromDDD {
public:
  ME0GeometryBuilderFromDDD();

  ~ME0GeometryBuilderFromDDD();

  ME0Geometry* build(const DDCompactView* cview, const MuonDDDConstants& muonConstants);

private:
  ME0Geometry* buildGeometry(DDFilteredView& fview, const MuonDDDConstants& muonConstants);
  std::map<ME0DetId, std::vector<ME0DetId>> chids;

  typedef ReferenceCountingPointer<BoundPlane> ME0BoundPlane;

  ME0BoundPlane boundPlane(const DDFilteredView& fv, Bounds* bounds, bool isOddChamber) const;

  ME0Chamber* buildChamber(DDFilteredView& fv, ME0DetId detId) const;

  ME0Layer* buildLayer(DDFilteredView& fv, ME0DetId detId) const;

  ME0EtaPartition* buildEtaPartition(DDFilteredView& fv, ME0DetId detId) const;
};

#endif
