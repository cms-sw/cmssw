/****************************************************************************
*
* This is a part of TOTEM offline software.
* Author:
*   Laurent Forthomme
*
****************************************************************************/

#ifndef Geometry_ForwardGeometry_TotemT2Tile_h
#define Geometry_ForwardGeometry_TotemT2Tile_h

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDesc.h"

#include <DD4hep/DD4hepUnits.h>

class TotemT2Tile {
public:
  TotemT2Tile();
  explicit TotemT2Tile(const DetGeomDesc*);
  ~TotemT2Tile();

  const GlobalPoint& centre() const { return centre_; }

private:
  dd4hep::PlacedVolume vol_;
  GlobalPoint centre_;
};

#endif
