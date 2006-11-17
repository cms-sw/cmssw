/** Implementation of the Model for RPC Geometry
 *
 *  \author M. Maggi - INFN Bari
 */

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include <Geometry/CommonDetUnit/interface/GeomDetUnit.h>
#include <Geometry/CommonDetUnit/interface/GeomDetType.h>


RPCGeometry::RPCGeometry(){}


RPCGeometry::~RPCGeometry()
{
  // delete all the roll associated to the geometry
  for (std::vector<RPCRoll*>::const_iterator iroll = allRolls.begin();
       iroll != allRolls.end(); ++iroll){
    delete (*iroll);
  }
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
  return theRollIds;
}


const GeomDetUnit* RPCGeometry::idToDetUnit(DetId id) const{
  if (idtoRoll.find(id) != idtoRoll.end())
    return (idtoRoll.find(id))->second;
  else
    return 0;
}


const GeomDet* RPCGeometry::idToDet(DetId id) const{
//   if (idtoDet.find(id) != idtoDet.end())
//     return (idtoDet.find(id))->second;
//   else
//     return 0;

  // For the time being, the only RPC dets are the rolls...
  return idToDetUnit(id);
}


const std::vector<RPCRoll*>& RPCGeometry::rolls() const{
  return allRolls;
}



const RPCRoll* RPCGeometry::roll(RPCDetId id) const{
  return (const RPCRoll*)(idToDetUnit(id));
}





void
RPCGeometry::add(RPCRoll* roll){
  allRolls.push_back(roll);
  theRolls.push_back(roll);
  theRollIds.push_back(roll->geographicalId());
  GeomDetType* _t = const_cast<GeomDetType*>(&roll->type());
  theRollTypes.push_back(_t);
  idtoRoll.insert(std::pair<DetId,GeomDetUnit*>
		  (roll->geographicalId(),roll));
  theDets.push_back(roll);
}

