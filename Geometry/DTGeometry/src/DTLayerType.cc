/** \file
 *
 *  \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 */

/* This Class Header */
#include "Geometry/DTGeometry/interface/DTLayerType.h"

/* Collaborating Class Header */
#include "Geometry/DTGeometry/interface/DTTopology.h"

/* Constructor */
DTLayerType::DTLayerType() : GeomDetType("DT", GeomDetEnumerators::DT) {}

/* Operations */
const Topology& DTLayerType::topology() const {
  static const DTTopology result(0, 0, 0.);
  return result;
}
