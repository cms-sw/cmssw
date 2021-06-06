#ifndef Geometry_GEMGeometry_ME0GeometryParsFromDD_H
#define Geometry_GEMGeometry_ME0GeometryParsFromDD_H

/* Implementation of the  ME0GeometryParsFromDD Class
 *  Build the ME0Geometry from the DDD and DD4Hep description
 *  
 *  DD4hep part added to the original old file (DD version) made by M. Maggi (INFN Bari)
 *  Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) 
 *  Created:  Thu, 25 Feb 2021 
 *
 */

#include <vector>

class DDCompactView;
class DDFilteredView;
namespace cms {  // DD4Hep
  class DDFilteredView;
  class DDCompactView;
}  // namespace cms
class MuonGeometryConstants;
class RecoIdealGeometry;
class ME0DetId;

class ME0GeometryParsFromDD {
public:
  ME0GeometryParsFromDD(void) {}

  ~ME0GeometryParsFromDD(void) {}
  // DD
  void build(const DDCompactView*, const MuonGeometryConstants&, RecoIdealGeometry&);
  // DD4HEP
  void build(const cms::DDCompactView*, const MuonGeometryConstants&, RecoIdealGeometry&);

private:
  // DD
  void buildGeometry(DDFilteredView&, const MuonGeometryConstants&, RecoIdealGeometry&);

  void buildChamber(DDFilteredView& fv, ME0DetId detId, RecoIdealGeometry& rgeo);
  void buildLayer(DDFilteredView& fv, ME0DetId detId, RecoIdealGeometry& rgeo);
  void buildEtaPartition(DDFilteredView& fv, ME0DetId detId, RecoIdealGeometry& rgeo);

  std::vector<double> getDimension(DDFilteredView& fv);
  std::vector<double> getTranslation(DDFilteredView& fv);
  std::vector<double> getRotation(DDFilteredView& fv);

  //DD4HEP

  void buildGeometry(cms::DDFilteredView&, const MuonGeometryConstants&, RecoIdealGeometry&);

  void buildChamber(cms::DDFilteredView& fv, ME0DetId detId, RecoIdealGeometry& rgeo);
  void buildLayer(cms::DDFilteredView& fv, ME0DetId detId, RecoIdealGeometry& rgeo);
  void buildEtaPartition(cms::DDFilteredView& fv, ME0DetId detId, RecoIdealGeometry& rgeo);

  std::vector<double> getDimension(cms::DDFilteredView& fv);
  std::vector<double> getTranslation(cms::DDFilteredView& fv);
  std::vector<double> getRotation(cms::DDFilteredView& fv);
};
#endif
