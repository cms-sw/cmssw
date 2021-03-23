#ifndef Geometry_GEMGeometry_GEMGeometryBuilder_H
#define Geometry_GEMGeometry_GEMGeometryBuilder_H

#include "DataFormats/GeometrySurface/interface/Plane.h"
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
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DD4hep/DD4hepUnits.h"

class DDCompactView;
class DDFilteredView;
namespace cms {
  class DDFilteredView;
  class DDCompactView;
  class MuonNumbering;
}  // namespace cms
class GEMGeometry;
class GEMSuperChamber;
class GEMChamber;
class GEMEtaPartition;
class MuonGeometryConstants;

class GEMGeometryBuilder {
public:
  GEMGeometryBuilder();

  ~GEMGeometryBuilder();

  // for DDD
  void build(GEMGeometry& theGeometry, const DDCompactView* cview, const MuonGeometryConstants& muonConstants);
  // for DD4hep
  void build(GEMGeometry& theGeometry, const cms::DDCompactView* cview, const MuonGeometryConstants& muonConstants);

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

  // Common
  void buildRegions(GEMGeometry&, const std::vector<GEMSuperChamber*>&);

  static constexpr double k_ScaleFromDD4Hep = (1.0 / dd4hep::cm);
};

#endif
