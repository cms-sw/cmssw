/** \file 
 *  
 *  $date   : 13/01/2006 11:46:51 CET $
 *  $Revision: $
 *  \author Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 */

/* This Class Header */

/* Base Class Headers */
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"

/* Collaborating Class Header */
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"

/* C++ Headers */
#include <iostream>

/* ====================================================================== */

/* Constructor */ 
DTSuperLayer::DTSuperLayer(DTSuperLayerId id,
                           ReferenceCountingPointer<BoundPlane>& plane,
                           const DTChamber* ch) :
  GeomDet(plane), theId(id) , theCh(ch){
}

/* Destructor */ 
DTSuperLayer::~DTSuperLayer() {
}

/* Operations */ 
DetId DTSuperLayer::geographicalId() const {
  return theId;
}

DTSuperLayerId DTSuperLayer::id() const {
  return theId;
}

bool DTSuperLayer::operator==(const DTSuperLayer& sl) const {
  return id()==sl.id();
}

/// Return the layers in the SL
std::vector< const GeomDet*> DTSuperLayer::components() const {
  std::vector<const GeomDet*> result;
  result.insert(result.end(), theLayers.begin(), theLayers.end());
  return result;
}

std::vector< const DTLayer*> DTSuperLayer::layers() const {
  return theLayers;
}

void DTSuperLayer::add(DTLayer* l) {
  theLayers.push_back(l);
}

const DTChamber* DTSuperLayer::chamber() const {
  return theCh;
}
