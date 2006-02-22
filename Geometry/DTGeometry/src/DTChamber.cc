/** \file
 *
 *  $Date: $
 *  $Revision: $
 *  \author Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 */

/* This Class Header */
#include "Geometry/DTGeometry/interface/DTChamber.h"

/* Collaborating Class Header */
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"

/* C++ Headers */
#include <iostream>

/* ====================================================================== */

/* Constructor */ 
DTChamber::DTChamber(DTChamberId id, const ReferenceCountingPointer<BoundPlane>& plane) :
  GeomDet(plane), 
  theId(id) {
}

/* Destructor */ 
DTChamber::~DTChamber() {
}

/* Operations */ 
DetId DTChamber::geographicalId() const {
  return theId;
}

DTChamberId DTChamber::id() const {
  return theId;
}

bool DTChamber::operator==(const DTChamber& ch) const {
  return id()==ch.id();
}

void DTChamber::add(DTSuperLayer* sl) {
  theSLs.push_back(sl);
}

std::vector<const GeomDet*> DTChamber::components() const {
  std::vector<const GeomDet*> result;
  result.insert(result.end(), theSLs.begin(), theSLs.end());
  // for (std::vector<const DTSuperLayer*>::iterator isl=theSLs.begin();
  //      isl!=theSLs.end(); ++isl) result.push_back(*isl);
  return result;
}

std::vector<const DTSuperLayer*> DTChamber::superLayers() const {
  return theSLs;
}
