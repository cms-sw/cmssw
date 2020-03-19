#ifndef Geometry_GEMGeometry_GEMGeometryParsFromDD_H
#define Geometry_GEMGeometry_GEMGeometryParsFromDD_H

/** \class  GEMGeometryParsFromDD
 *  Build the GEMGeometry ftom the DDD description
 *
 *  \author M. Maggi - INFN Bari
 *
 */

#include <string>
#include <vector>
#include <map>
#include <list>

class DDCompactView;
class DDFilteredView;
class MuonDDDConstants;
class RecoIdealGeometry;
class GEMDetId;

class GEMGeometryParsFromDD {
public:
  GEMGeometryParsFromDD();

  ~GEMGeometryParsFromDD();

  void build(const DDCompactView* cview, const MuonDDDConstants& muonConstants, RecoIdealGeometry& rgeo);

private:
  void buildGeometry(DDFilteredView& fview, const MuonDDDConstants& muonConstants, RecoIdealGeometry& rgeo);

  void buildSuperChamber(DDFilteredView& fv, GEMDetId detId, RecoIdealGeometry& rgeo);
  void buildChamber(DDFilteredView& fv, GEMDetId detId, RecoIdealGeometry& rgeo);
  void buildEtaPartition(DDFilteredView& fv, GEMDetId detId, RecoIdealGeometry& rgeo);

  std::vector<double> getTranslation(DDFilteredView& fv);
  std::vector<double> getRotation(DDFilteredView& fv);
};
#endif
