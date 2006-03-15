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
  void buildPixelBarrel(std::vector<const GeometricDet*>,DDExpandedView*,TrackerGeometry*,GeomDetType::SubDetector& det);
  void buildPixelForward(std::vector<const GeometricDet*>,DDExpandedView*,TrackerGeometry*,GeomDetType::SubDetector& det);
  void buildSiliconBarrel(std::vector<const GeometricDet*>,DDExpandedView*,TrackerGeometry*,GeomDetType::SubDetector& det);
  void buildSiliconForward(std::vector<const GeometricDet*>,DDExpandedView*,TrackerGeometry*,GeomDetType::SubDetector& det);
  void buildGeomDet(TrackerGeometry*);
  std::string getString(std::string,DDExpandedView*);
  double getDouble(std::string,DDExpandedView*);

  GeomDetTypeIdToEnum theDetIdToEnum;
  GeomTopologyBuilder* theTopologyBuilder;

};

#endif
