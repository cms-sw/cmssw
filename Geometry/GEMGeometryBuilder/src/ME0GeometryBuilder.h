#ifndef Geometry_GEMGeometry_ME0GeometryBuilder_H
#define Geometry_GEMGeometry_ME0GeometryBuilder_H
/*
//\class ME0GeometryBuilder

 Description: ME0 Geometry builder for DD4HEP
              DD4hep part added to the original old file (DD version) made by M. Maggi (INFN Bari)
//
// Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osborne made for DTs (DD4HEP migration)
//          Created:  29 Apr 2019 
*/

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DD4hep/DD4hepUnits.h"
#include <string>
#include <map>
#include <vector>

class DDCompactView;
class DDFilteredView;
namespace cms {
  class DDFilteredView;
  class DDCompactView;
}  // namespace cms
class ME0Geometry;
class ME0Chamber;
class ME0Layer;
class ME0EtaPartition;
class MuonGeometryConstants;

class ME0GeometryBuilder {
public:
  ME0GeometryBuilder();

  ~ME0GeometryBuilder();

  ME0Geometry* build(const DDCompactView* cview, const MuonGeometryConstants& muonConstants);
  //dd4hep
  ME0Geometry* build(const cms::DDCompactView* cview, const MuonGeometryConstants& muonConstants);

private:
  ME0Geometry* buildGeometry(DDFilteredView& fview, const MuonGeometryConstants& muonConstants);
  std::map<ME0DetId, std::vector<ME0DetId>> chids;

  typedef ReferenceCountingPointer<BoundPlane> ME0BoundPlane;

  ME0BoundPlane boundPlane(const DDFilteredView& fv, Bounds* bounds, bool isOddChamber) const;

  ME0Chamber* buildChamber(DDFilteredView& fv, ME0DetId detId) const;

  ME0Layer* buildLayer(DDFilteredView& fv, ME0DetId detId) const;

  ME0EtaPartition* buildEtaPartition(DDFilteredView& fv, ME0DetId detId) const;
  //dd4hep
  ME0Geometry* buildGeometry(cms::DDFilteredView& fview, const MuonGeometryConstants& muonConstants);

  ME0BoundPlane boundPlane(const cms::DDFilteredView& fv, Bounds* bounds, bool isOddChamber) const;

  ME0Chamber* buildChamber(cms::DDFilteredView& fv, ME0DetId detId) const;

  ME0Layer* buildLayer(cms::DDFilteredView& fv, ME0DetId detId) const;

  ME0EtaPartition* buildEtaPartition(cms::DDFilteredView& fv, ME0DetId detId) const;

  static constexpr double k_ScaleFromDD4Hep = (1.0 / dd4hep::cm);
};

#endif
