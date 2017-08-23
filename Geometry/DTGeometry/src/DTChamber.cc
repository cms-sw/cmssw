/** \file
 *
 *  \author Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 */

/* This Class Header */
#include "Geometry/DTGeometry/interface/DTChamber.h"

/* Collaborating Class Header */
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

/* C++ Headers */
#include <iostream>

/* ====================================================================== */

/* Constructor */ 
DTChamber::DTChamber(const DTChamberId& id, const ReferenceCountingPointer<BoundPlane>& plane) :
  GeomDet(plane), 
  theId(id) {
  setDetId(id);
}

/* Destructor */ 
DTChamber::~DTChamber() {
  for (std::vector<const DTSuperLayer*>::const_iterator isl=theSLs.begin();
       isl!=theSLs.end(); ++isl) delete (*isl);
}

/* Operations */ 

DTChamberId DTChamber::id() const {
  return theId;
}

bool DTChamber::operator==(const DTChamber& ch) const {
  return id()==ch.id();
}

void DTChamber::add(DTSuperLayer* sl) {
  theSLs.emplace_back(sl);
}

std::vector<const GeomDet*> DTChamber::components() const {
  return  std::vector<const GeomDet*>(theSLs.begin(), theSLs.end());
}


const GeomDet* DTChamber::component(DetId id) const {
  DTLayerId lId(id.rawId());
  if (lId.layer()==0) { // is a SL id
    return superLayer(lId);
  } else { // is a layer id
    return layer(lId);
  }
}


const std::vector<const DTSuperLayer*>& DTChamber::superLayers() const {
  return theSLs;
}


const DTSuperLayer* DTChamber::superLayer(const DTSuperLayerId& id) const{
  if (id.chamberId()!=theId) return nullptr; // not in this SL!
  return superLayer(id.superLayer());
}


const DTSuperLayer* DTChamber::superLayer(int isl) const {
  for (auto theSL : theSLs) {
    if (theSL->id().superLayer()==isl) return theSL;
  }
  return nullptr;
}


const DTLayer* DTChamber::layer(const DTLayerId& id) const {
  return (superLayer(id.superlayer()))->layer(id.layer());
}

