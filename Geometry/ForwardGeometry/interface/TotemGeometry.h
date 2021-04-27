/****************************************************************************
*
* This is a part of TOTEM offline software.
* Author:
*   Laurent Forthomme
*
****************************************************************************/

#ifndef Geometry_ForwardGeometry_TotemGeometry_h
#define Geometry_ForwardGeometry_TotemGeometry_h

#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDesc.h"

#include "DataFormats/CTPPSDetId/interface/TotemT2DetId.h"

class TotemGeometry {
public:
  TotemGeometry(const DetGeomDesc*);

  bool addT2Sector(const TotemT2DetId&, const DetGeomDesc*&);
  bool addT2Plane(const TotemT2DetId&, const DetGeomDesc*&);
  bool addT2Tile(const TotemT2DetId&, const DetGeomDesc*&);

  const DetGeomDesc*& tile(const TotemT2DetId&) const;

private:
  std::map<CTPPSDetId, const DetGeomDesc*&> nt2_tiles_;
};

#endif
