#ifndef Geometry_TrackerGeometryBuilder_TrackerGeomBuilderFromGeometricDet_H
#define Geometry_TrackerGeometryBuilder_TrackerGeomBuilderFromGeometricDet_H

#include<string>
#include<vector>
#include "Geometry/TrackerGeometryBuilder/interface/PlaneBuilderFromGeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/GeomDetTypeIdToEnum.h"
#include "Geometry/TrackerGeometryBuilder/interface/GeomTopologyBuilder.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

class TrackerGeometry;


class TrackerGeomBuilderFromGeometricDet {
public:

  TrackerGeometry* build(const DDCompactView* fv, const GeometricDet* gd);

private:

  void buildPixel(std::vector<const GeometricDet*> const &,DDExpandedView*,
		  TrackerGeometry*,GeomDetType::SubDetector& det, const std::string& part);
  void buildSilicon(std::vector<const GeometricDet*> const &,DDExpandedView*,
		    TrackerGeometry*,GeomDetType::SubDetector& det, const std::string& part);
  void buildGeomDet(TrackerGeometry*);

  std::string getString(std::string const &, DDExpandedView*) const;
  double getDouble(std::string const &, DDExpandedView*) const;

  PlaneBuilderFromGeometricDet::ResultType
  buildPlaneWithMaterial(const GeometricDet* gd) const;

  GeomDetTypeIdToEnum theDetIdToEnum;
  GeomTopologyBuilder* theTopologyBuilder;

};

#endif
