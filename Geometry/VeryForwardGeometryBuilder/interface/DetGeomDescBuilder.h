#ifndef Geometry_VeryForwardGeometryBuilder_DetGeomDescBuilder
#define Geometry_VeryForwardGeometryBuilder_DetGeomDescBuilder

#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDesc.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"

/*
 * Generic function to build geo (tree of DetGeomDesc) from compact view.
 */
namespace detgeomdescbuilder {
  std::unique_ptr<DetGeomDesc> buildDetGeomDescFromCompactView(const DDCompactView& myCompactView);
  void buildDetGeomDescDescendants(DDFilteredView& fv, DetGeomDesc* geoInfoParent);
  std::unique_ptr<DetGeomDesc> buildDetGeomDescFromCompactView(const cms::DDCompactView& myCompactView);
}  // namespace detgeomdescbuilder

#endif
