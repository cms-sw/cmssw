/** \file
 *
 *  $Date: 2006/09/21 15:33:47 $
 *  $Revision: 1.1 $
 *  \author Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 */

/* This Class Header */
#include "Geometry/RPCGeometry/interface/RPCChamber.h"

/* Collaborating Class Header */
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

/* C++ Headers */
#include <iostream>

/* ====================================================================== */

/* Constructor */ 
RPCChamber::RPCChamber(RPCDetId id, const BoundPlane::BoundPlanePointer& plane) :
  GeomDet(plane), 
  theId(id) {
}

/* Destructor */ 
RPCChamber::~RPCChamber() {
    // delete all rolls associated to this chamber
  for (std::vector<const RPCRoll*>::const_iterator iroll = theRolls.begin();
       iroll != theRolls.end(); ++iroll){
    delete (*iroll);
  }
}

/* Operations */ 
DetId RPCChamber::geographicalId() const {
  return theId;
}

RPCDetId RPCChamber::id() const {
  return theId;
}

bool RPCChamber::operator==(const RPCChamber& ch) const {
  return id()==ch.id();
}

void RPCChamber::add(RPCRoll* rl) {
  theRolls.push_back(rl);
}

std::vector<const GeomDet*> RPCChamber::components() const {
  return  std::vector<const GeomDet*>(theRolls.begin(), theRolls.end());
}

const GeomDet* RPCChamber::component(DetId id) const {
  return roll(RPCDetId(id.rawId()));
}

const std::vector<const RPCRoll*>& RPCChamber::rolls() const {
  return theRolls;
}

const RPCRoll* RPCChamber::roll(RPCDetId id) const{
  if (id.chamberId()!=theId) return 0; // not in this Roll!
  return roll(id.roll());
}


const RPCRoll* RPCChamber::roll(int isl) const {
  for (std::vector<const RPCRoll*>::const_iterator i = theRolls.begin();
       i!= theRolls.end(); ++i) {
    if ((*i)->id().roll()==isl) return (*i);
  }
  return 0;
}

