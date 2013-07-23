/** \file
 *
 *  $Date: 2010/04/09 12:17:26 $
 *  $Revision: 1.7 $
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
DTChamber::DTChamber(DTChamberId id, const ReferenceCountingPointer<BoundPlane>& plane) :
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
  theSLs.push_back(sl);
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


const DTSuperLayer* DTChamber::superLayer(DTSuperLayerId id) const{
  if (id.chamberId()!=theId) return 0; // not in this SL!
  return superLayer(id.superLayer());
}


const DTSuperLayer* DTChamber::superLayer(int isl) const {
  for (std::vector<const DTSuperLayer*>::const_iterator i = theSLs.begin();
       i!= theSLs.end(); ++i) {
    if ((*i)->id().superLayer()==isl) return (*i);
  }
  return 0;
}


const DTLayer* DTChamber::layer(DTLayerId id) const {
  return (superLayer(id.superlayer()))->layer(id.layer());
}

