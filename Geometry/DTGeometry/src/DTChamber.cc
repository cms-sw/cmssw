/** \file
 *
 *  $Date: 2006/05/31 17:13:26 $
 *  $Revision: 1.3 $
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
  return  std::vector<const GeomDet*>(theSLs.begin(), theSLs.end());
}

const std::vector<const DTSuperLayer*>& DTChamber::superLayers() const {
  return theSLs;
}


const DTSuperLayer* DTChamber::superLayer(DTSuperLayerId id) const{
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

