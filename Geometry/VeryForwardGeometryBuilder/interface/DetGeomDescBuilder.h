#ifndef Geometry_VeryForwardGeometryBuilder_DetGeomDescBuilder
#define Geometry_VeryForwardGeometryBuilder_DetGeomDescBuilder

#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDesc.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"


/*
 * Generic function to build geo (tree of DetGeomDesc) from compact view.
 */
namespace DetGeomDescBuilder {
  std::unique_ptr<DetGeomDesc> buildDetGeomDescFromCompactView(const cms::DDCompactView& myCompactView);
}

#endif
