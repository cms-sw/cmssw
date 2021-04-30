/****************************************************************************
*
* This is a part of TOTEM offline software.
* Author:
*   Laurent Forthomme
*
****************************************************************************/

#include "Geometry/ForwardGeometry/interface/TotemGeometry.h"

TotemGeometry::TotemGeometry(const DetGeomDesc* dgd) {
  std::deque<const DetGeomDesc*> buffer;
  buffer.emplace_back(dgd);
  while (!buffer.empty()) {
    // get the next item
    const auto* desc = buffer.front();
    buffer.pop_front();

    desc->print();
  }
}

bool TotemGeometry::addT2Sector(const TotemT2DetId&, const DetGeomDesc*&) { return true; }

bool TotemGeometry::addT2Plane(const TotemT2DetId&, const DetGeomDesc*&) { return true; }

bool TotemGeometry::addT2Tile(const TotemT2DetId&, const DetGeomDesc*&) { return true; }

const DetGeomDesc*& TotemGeometry::tile(const TotemT2DetId& detid) const { return nt2_tiles_.at(detid); }
