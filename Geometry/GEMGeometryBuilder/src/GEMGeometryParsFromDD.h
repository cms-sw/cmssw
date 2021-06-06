#ifndef Geometry_GEMGeometry_GEMGeometryParsFromDD_H
#define Geometry_GEMGeometry_GEMGeometryParsFromDD_H

/* Implementation of the  GEMGeometryParsFromDD Class
 *  Build the GEMGeometry from the DDD and DD4Hep description
 *  
 *  DD4hep part added to the original old file (DD version) made by M. Maggi (INFN Bari)
 *  Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) 
 *  Created:  Mon, 15 Feb 2021 
 *
 */

#include <string>
#include <vector>
#include <map>
#include <list>

class DDCompactView;
class DDFilteredView;
namespace cms {  // DD4Hep
  class DDFilteredView;
  class DDCompactView;
}  // namespace cms
class MuonGeometryConstants;
class RecoIdealGeometry;
class GEMDetId;

class GEMGeometryParsFromDD {
public:
  GEMGeometryParsFromDD();

  ~GEMGeometryParsFromDD();

  // DD
  void build(const DDCompactView* cview, const MuonGeometryConstants& muonConstants, RecoIdealGeometry& rgeo);
  // DD4Hep
  void build(const cms::DDCompactView* cview, const MuonGeometryConstants& muonConstants, RecoIdealGeometry& rgeo);

private:
  // DD
  void buildGeometry(DDFilteredView& fview, const MuonGeometryConstants& muonConstants, RecoIdealGeometry& rgeo);
  void buildSuperChamber(DDFilteredView& fv, GEMDetId detId, RecoIdealGeometry& rgeo);
  void buildChamber(DDFilteredView& fv, GEMDetId detId, RecoIdealGeometry& rgeo);
  void buildEtaPartition(DDFilteredView& fv, GEMDetId detId, RecoIdealGeometry& rgeo);

  std::vector<double> getTranslation(DDFilteredView& fv);
  std::vector<double> getRotation(DDFilteredView& fv);

  // DD4Hep

  void buildGeometry(cms::DDFilteredView& fview, const MuonGeometryConstants& muonConstants, RecoIdealGeometry& rgeo);
  void buildSuperChamber(cms::DDFilteredView& fv, GEMDetId detId, RecoIdealGeometry& rgeo);
  void buildChamber(cms::DDFilteredView& fv, GEMDetId detId, RecoIdealGeometry& rgeo);
  void buildEtaPartition(cms::DDFilteredView& fv, GEMDetId detId, RecoIdealGeometry& rgeo);

  std::vector<double> getTranslation(cms::DDFilteredView& fv);
  std::vector<double> getRotation(cms::DDFilteredView& fv);
};
#endif
