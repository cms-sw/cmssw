#ifndef Geometry_GEMGeometry_GEMGeometryBuilderFromDDD_H
#define Geometry_GEMGeometry_GEMGeometryBuilderFromDDD_H

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "Geometry/MuonNumbering/interface/DD4hep_GEMNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"
/*
//\class GEMGeometryBuilder

 Description: GEM Geometry builder from DD & DD4HEP
              DD4hep part added to the original old file (DD version) made by M. Maggi (INFN Bari)
//
// Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osburne made for DTs (DD4HEP migration)
//          Created:  27 Jan 2020 
*/
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
class GEMGeometry;
class GEMDetId;
class GEMSuperChamber;
class GEMChamber;
class GEMEtaPartition;
class MuonDDDConstants;

class GEMGeometryBuilderFromDDD {
public:
  GEMGeometryBuilderFromDDD();

  ~GEMGeometryBuilderFromDDD();

  // for DDD
  void build(GEMGeometry& theGeometry, const DDCompactView* cview, const MuonDDDConstants& muonConstants);
  // for DD4hep
  void build(GEMGeometry& theGeometry, const cms::DDCompactView* cview, const cms::MuonNumbering& muonConstants);

private:
  std::map<GEMDetId, std::vector<GEMDetId>> chids;

  // for DDD
  typedef ReferenceCountingPointer<BoundPlane> RCPBoundPlane;

  RCPBoundPlane boundPlane(const DDFilteredView& fv, Bounds* bounds, bool isOddChamber) const;

  GEMSuperChamber* buildSuperChamber(DDFilteredView& fv, GEMDetId detId) const;

  GEMChamber* buildChamber(DDFilteredView& fv, GEMDetId detId) const;

  GEMEtaPartition* buildEtaPartition(DDFilteredView& fv, GEMDetId detId) const;

  // for DD4hep

  RCPBoundPlane boundPlane(const cms::DDFilteredView& fv, Bounds* bounds, bool isOddChamber) const;

  GEMSuperChamber* buildSuperChamber(cms::DDFilteredView& fv, GEMDetId detId) const;

  GEMChamber* buildChamber(cms::DDFilteredView& fv, GEMDetId detId) const;

  GEMEtaPartition* buildEtaPartition(cms::DDFilteredView& fv, GEMDetId detId) const;
};

#endif
