#ifndef Geometry_GEMGeometry_ME0GeometryParsFromDD_H
#define Geometry_GEMGeometry_ME0GeometryParsFromDD_H

#include <vector>

class DDCompactView;
class DDFilteredView;
class MuonDDDConstants;
class RecoIdealGeometry;
class ME0DetId;

class ME0GeometryParsFromDD {
public:
  ME0GeometryParsFromDD(void) {}

  ~ME0GeometryParsFromDD(void) {}

  void build(const DDCompactView*, const MuonDDDConstants&, RecoIdealGeometry&);

private:
  void buildGeometry(DDFilteredView&, const MuonDDDConstants&, RecoIdealGeometry&);

  void buildChamber(DDFilteredView& fv, ME0DetId detId, RecoIdealGeometry& rgeo);
  void buildLayer(DDFilteredView& fv, ME0DetId detId, RecoIdealGeometry& rgeo);
  void buildEtaPartition(DDFilteredView& fv, ME0DetId detId, RecoIdealGeometry& rgeo);

  std::vector<double> getDimension(DDFilteredView& fv);
  std::vector<double> getTranslation(DDFilteredView& fv);
  std::vector<double> getRotation(DDFilteredView& fv);
};
#endif
