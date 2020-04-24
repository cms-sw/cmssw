/** \file
 *
 *  Author: Raffaello Trentadue - Universit? Bari 
 *  Mail:     <raffaello.trentadue@ba.infn.it>
 */

/* This Class Header */
#include "Geometry/RPCGeometry/interface/RPCChamber.h"

/* Collaborating Class Header */
#include "Geometry/RPCGeometry/interface/RPCRoll.h"


/* C++ Headers */
#include <iostream>

/* ====================================================================== */

/* Constructor */ 
RPCChamber::RPCChamber(RPCDetId id, 
		       const ReferenceCountingPointer<BoundPlane> & plane) :
  GeomDet(plane), theId(id)
{
  setDetId(id);
}

/* Destructor */ 
RPCChamber::~RPCChamber() {
}


RPCDetId
RPCChamber::id() const
{
  return theId;
}

/* Operations */ 

bool 
RPCChamber::operator==(const RPCChamber& ch) const {
  return this->id()==ch.id();
}



void 
RPCChamber::add(RPCRoll* rl) {
  theRolls.emplace_back(rl);
}



std::vector<const GeomDet*> 
RPCChamber::components() const {
  return  std::vector<const GeomDet*>(theRolls.begin(), theRolls.end());
}



const GeomDet* 
RPCChamber::component(DetId id) const {
  return roll(RPCDetId(id.rawId()));
}


const std::vector<const RPCRoll*>& 
RPCChamber::rolls() const 
{
  return theRolls;
}

int
RPCChamber::nrolls() const
{
  return theRolls.size();
}

const RPCRoll* 
RPCChamber::roll(RPCDetId id) const
{
  if (id.chamberId()!=theId) return nullptr; // not in this Roll!
  return roll(id.roll());
}


const RPCRoll* 
RPCChamber::roll(int isl) const 
{
  for (auto theRoll : theRolls) {
    if (theRoll->id().roll()==isl) 
      return theRoll;
  }
  return nullptr;
}

