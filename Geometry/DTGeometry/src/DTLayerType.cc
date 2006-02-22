/** \file
 *
 *  $Revision: $
 *  $date   : 23/01/2006 18:24:56 CET $
 *  \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 */

/* This Class Header */
#include "Geometry/DTGeometry/interface/DTLayerType.h"

/* Collaborating Class Header */
#include "Geometry/DTGeometry/interface/DTTopology.h"

/* Base Class Headers */

/* C++ Headers */

/* ====================================================================== */

/* Constructor */ 
DTLayerType::DTLayerType() :
  GeomDetType("DT",DT){
  }

/* Destructor */ 

/* Operations */ 
const Topology& DTLayerType::topology() const {
  static DTTopology result(0,0,0.);
  return result;
}


