/** Implementation of the Model for RPC Geometry
 *
 *  \author M. Maggi - INFN Bari
 */

#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/CommonDetUnit/interface/GeomDetUnit.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

RPCGeometry::RPCGeometry(){}


RPCGeometry::~RPCGeometry()
{
  // delete all the chamber associated to the geometry
  //for (std::vector<RPCChamber*>::const_iterator ich = allChambers.begin();
  //      ich != allChambers.end(); ++ich){
  //  delete (*ich);
  //}
}  

 
const RPCGeometry::DetTypeContainer&  RPCGeometry::detTypes() const{
  return theRollTypes;
}


const RPCGeometry::DetUnitContainer& RPCGeometry::detUnits() const{
  return theRolls;
}


const RPCGeometry::DetContainer& RPCGeometry::dets() const{
  return theDets;
}

  
const RPCGeometry::DetIdContainer& RPCGeometry::detUnitIds() const{
  return theRollIds;
}


const RPCGeometry::DetIdContainer& RPCGeometry::detIds() const{
  return theDetIds;
}


const GeomDetUnit* RPCGeometry::idToDetUnit(DetId id) const{
  return dynamic_cast<const GeomDetUnit*>(idToDet(id));
}


const GeomDet* RPCGeometry::idToDet(DetId id) const{
  mapIdToDet::const_iterator i = theMap.find(id);
  if (i != theMap.end())
    return i->second;

  LogDebug("RPCGeometry")<<"Invalid DetID: no GeomDet associated "<< RPCDetId(id);
  GeomDet* geom = 0;
  return geom;   
}

const std::vector<RPCChamber*>& RPCGeometry::chambers() const {
  return allChambers;
}

const std::vector<RPCRoll*>& RPCGeometry::rolls() const{
  return allRolls;
}

const RPCChamber* RPCGeometry::chamber(RPCDetId id) const{
  return dynamic_cast<const RPCChamber*>(idToDet(id.chamberId()));
}

const RPCRoll* RPCGeometry::roll(RPCDetId id) const{
  return dynamic_cast<const RPCRoll*>(idToDetUnit(id));
}


void
RPCGeometry::add(RPCRoll* roll){
  theDets.push_back(roll);
  allRolls.push_back(roll);
  theRolls.push_back(roll);
  theRollIds.push_back(roll->geographicalId());
  theDetIds.push_back(roll->geographicalId());
  GeomDetType* _t = const_cast<GeomDetType*>(&roll->type());
  theRollTypes.push_back(_t);
  theMap.insert(std::pair<DetId,GeomDetUnit*>
		(roll->geographicalId(),roll));
}

void
RPCGeometry::add(RPCChamber* chamber){
  allChambers.push_back(chamber);
  theDets.push_back(chamber);
  theDetIds.push_back(chamber->geographicalId());
  theMap.insert(std::pair<DetId,GeomDet*>
		(chamber->geographicalId(),chamber));
}
