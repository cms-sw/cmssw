#ifndef Geometry_GEMGeometry_ME0GeometryBuilderFromDDD_H
#define Geometry_GEMGeometry_ME0GeometryBuilderFromDDD_H
/*
//\class ME0GeometryBuilder

 Description: ME0 Geometry builder for DD4HEP
              DD4hep part added to the original old file (DD version) made by M. Maggi (INFN Bari)
//
// Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osborne made for DTs (DD4HEP migration)
//          Created:  29 Apr 2019 
*/

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include <string>
#include <map>
#include <vector>

class DDCompactView;
class DDFilteredView;
namespace cms {
  class DDFilteredView;
  class DDCompactView;
  class MuonNumbering;
  struct DDSpecPar;
  struct DDSpecParRegistry;
}  // namespace cms
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
  //dd4hep
  ME0Geometry* build(const cms::DDCompactView* cview, const cms::MuonNumbering& muonConstants);

private:
  ME0Geometry* buildGeometry(DDFilteredView& fview, const MuonDDDConstants& muonConstants);
  std::map<ME0DetId, std::vector<ME0DetId>> chids;

  typedef ReferenceCountingPointer<BoundPlane> ME0BoundPlane;

  ME0BoundPlane boundPlane(const DDFilteredView& fv, Bounds* bounds, bool isOddChamber) const;

  ME0Chamber* buildChamber(DDFilteredView& fv, ME0DetId detId) const;

  ME0Layer* buildLayer(DDFilteredView& fv, ME0DetId detId) const;

  ME0EtaPartition* buildEtaPartition(DDFilteredView& fv, ME0DetId detId) const;
  //dd4hep
  ME0Geometry* buildGeometry(cms::DDFilteredView& fview, const cms::MuonNumbering& muonConstants);

  ME0BoundPlane boundPlane(const cms::DDFilteredView& fv, Bounds* bounds, bool isOddChamber) const;

  ME0Chamber* buildChamber(cms::DDFilteredView& fv, ME0DetId detId) const;

  ME0Layer* buildLayer(cms::DDFilteredView& fv, ME0DetId detId) const;

  ME0EtaPartition* buildEtaPartition(cms::DDFilteredView& fv, ME0DetId detId) const;
};

#endif
