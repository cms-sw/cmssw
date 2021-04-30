/****************************************************************************
*
* This is a part of TOTEM offline software.
* Author:
*   Laurent Forthomme
*
****************************************************************************/

#include "Geometry/ForwardGeometry/interface/TotemT2Tile.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"

TotemT2Tile::TotemT2Tile() {}

TotemT2Tile::TotemT2Tile(const DetGeomDesc* dgd) {
  centre_ = GlobalPoint{(float)dgd->translation().x(),
                        (float)dgd->translation().y(),
                        (float)dgd->parentZPosition()};  // retrieve the plane position for z coordinate
  const dd4hep::Volume box;                              //(0, 0, 0);
  const dd4hep::Trapezoid trap(
      dgd->params()[4], dgd->params()[8], dgd->params()[3], dgd->params()[3], dgd->params()[0]);
  const dd4hep::Transform3D tile_transform(dgd->rotation(), dgd->translation());
}

TotemT2Tile::~TotemT2Tile() {}
