#ifndef Geometry_TrackerGeometryBuilder_TrackerGeomBuilderFromGeometricDet_H
#define Geometry_TrackerGeometryBuilder_TrackerGeomBuilderFromGeometricDet_H

#include<string>
#include<vector>
#include <map>
#include <string>
#include "Geometry/TrackerGeometryBuilder/interface/PlaneBuilderFromGeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/GeomDetTypeIdToEnum.h"
#include "Geometry/TrackerGeometryBuilder/interface/GeomTopologyBuilder.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

class TrackerGeometry;
class PixelGeomDetType;
class StripGeomDetType;

class TrackerGeomBuilderFromGeometricDet {
public:

  TrackerGeometry* build(const GeometricDet* gd, bool upgradeGeometry,
			 int BIG_PIX_PER_ROC_X,
			 int BIG_PIX_PER_ROC_Y);
private:

  void buildPixel(std::vector<const GeometricDet*> const &,
		  TrackerGeometry*,GeomDetType::SubDetector det, const std::string& part,
		  bool upgradeGeometry,
		  int BIG_PIX_PER_ROC_X,
		  int BIG_PIX_PER_ROC_Y);
  void buildSilicon(std::vector<const GeometricDet*> const &,
		    TrackerGeometry*,GeomDetType::SubDetector det, const std::string& part);
  void buildGeomDet(TrackerGeometry*);

/*   std::string getString(std::string const &, DDExpandedView*) const; */
/*   double getDouble(std::string const &, DDExpandedView*) const; */

  PlaneBuilderFromGeometricDet::ResultType
  buildPlaneWithMaterial(const GeometricDet* gd,double scaleFactor=1.) const;

  GeomDetTypeIdToEnum theDetIdToEnum;
  GeomTopologyBuilder* theTopologyBuilder;

  std::map<std::string,PixelGeomDetType*> thePixelDetTypeMap;
  std::map<std::string,StripGeomDetType*> theStripDetTypeMap;
};

#endif
