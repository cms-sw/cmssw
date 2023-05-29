/****************************************************************************
*
* This is a part of TOTEM offline software.
* Author:
*   Laurent Forthomme
*
****************************************************************************/

#include "Geometry/ForwardGeometry/interface/TotemGeometry.h"

TotemGeometry::TotemGeometry(const DetGeomDesc* dgd) { browse(dgd, false); }

void TotemGeometry::browse(const DetGeomDesc* parent, bool in_t2) {
  if (parent->name() == "TotemT2")
    in_t2 = true;  // define the mother volume for all children
  if (in_t2)
    browseT2(parent);
  // start the recursive browsing
  for (const auto& child : parent->components())
    browse(child, in_t2);
}

void TotemGeometry::browseT2(const DetGeomDesc*& parent) {
  const unsigned short arm = parent->parentZPosition() > 0. ? 0 : 1;
  if (parent->name() == "TotemT2SupportBox")
    addT2Plane(TotemT2DetId(arm, parent->copyno() - 1), parent);
  else if (parent->name() == "TotemT2Scint") {
    unsigned short plane = 2 * (parent->copyno() / 10);
    unsigned short tile = parent->copyno() % 10;
    if (tile % 2 == 0)
      plane += 1;
    tile = (tile + 1) / 2 - 1;
    addT2Tile(TotemT2DetId(arm, plane, tile), parent);
  }
}

bool TotemGeometry::addT2Plane(const TotemT2DetId& detid, const DetGeomDesc*& dgd) {
  if (nt2_planes_.count(detid) != 0)
    return true;
  nt2_planes_[detid] = dgd;
  return true;
}

const DetGeomDesc* TotemGeometry::plane(const TotemT2DetId& detid) const { return nt2_planes_.at(detid); }

bool TotemGeometry::addT2Tile(const TotemT2DetId& detid, const DetGeomDesc*& dgd) {
  if (nt2_tiles_.count(detid) != 0)
    return false;
  nt2_tiles_[detid] = TotemT2Tile(dgd);
  return true;
}

const TotemT2Tile& TotemGeometry::tile(const TotemT2DetId& detid) const { return nt2_tiles_.at(detid); }
