/** \file 
 *  
 *  \author Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 */

/* This Class Header */

/* Base Class Headers */
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"

/* Collaborating Class Header */
#include "Geometry/DTGeometry/interface/DTLayer.h"

/* C++ Headers */
#include <iostream>

/* ====================================================================== */

/* Constructor */ 
DTSuperLayer::DTSuperLayer(const DTSuperLayerId& id,
                           ReferenceCountingPointer<BoundPlane>& plane,
                           const DTChamber* ch) :
  GeomDet(plane), theId(id) , theLayers(4,(const DTLayer*)nullptr), theCh(ch) {
  setDetId(id);
}

/* Destructor */ 
DTSuperLayer::~DTSuperLayer() {
  for (std::vector<const DTLayer*>::const_iterator il=theLayers.begin();
       il!=theLayers.end(); ++il) delete (*il);
}

/* Operations */ 

DTSuperLayerId DTSuperLayer::id() const {
  return theId;
}

bool DTSuperLayer::operator==(const DTSuperLayer& sl) const {
  return id()==sl.id();
}

/// Return the layers in the SL
std::vector< const GeomDet*> DTSuperLayer::components() const {
  return std::vector<const GeomDet*>(theLayers.begin(), theLayers.end());
}


const GeomDet* DTSuperLayer::component(DetId id) const {
  return layer(DTLayerId(id.rawId()));
}


const std::vector< const DTLayer*>& DTSuperLayer::layers() const {
  return theLayers;
}

void DTSuperLayer::add(DTLayer* l) {
  // theLayers size is preallocated.
  theLayers[l->id().layer()-1] = l;
}

const DTChamber* DTSuperLayer::chamber() const {
  return theCh;
}

const DTLayer* DTSuperLayer::layer(const DTLayerId& id) const {
  if (id.superlayerId()!=theId) return nullptr; // not in this SL!
  return layer(id.layer());
}
  
const DTLayer* DTSuperLayer::layer(int ilay) const{
  if ((ilay>=1) && (ilay<=4)) {
    return theLayers[ilay-1];
  } else {
    return nullptr;
  }
}
