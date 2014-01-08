/** Implementation of the Model for ME0 Geometry
 *
 *  \author M. Maggi - INFN Bari
 */

#include <Geometry/GEMGeometry/interface/ME0Geometry.h>
#include <Geometry/CommonDetUnit/interface/GeomDetUnit.h>

ME0Geometry::ME0Geometry(){}


ME0Geometry::~ME0Geometry(){}  

 
const ME0Geometry::DetTypeContainer&  ME0Geometry::detTypes() const{
  return theEtaPartitionTypes;
}


const ME0Geometry::DetUnitContainer& ME0Geometry::detUnits() const{
  return theEtaPartitions;
}


const ME0Geometry::DetContainer& ME0Geometry::dets() const{
  return theDets;
}

  
const ME0Geometry::DetIdContainer& ME0Geometry::detUnitIds() const{
  return theEtaPartitionIds;
}


const ME0Geometry::DetIdContainer& ME0Geometry::detIds() const{
  return theDetIds;
}


const GeomDetUnit* ME0Geometry::idToDetUnit(DetId id) const{
  return dynamic_cast<const GeomDetUnit*>(idToDet(id));
}

const GeomDet* ME0Geometry::idToDet(DetId id) const{
  mapIdToDet::const_iterator i = theMap.find(id);
  return (i != theMap.end()) ?
    i->second : 0 ;
}

/*
const std::vector<ME0Chamber*>& ME0Geometry::chambers() const {
  return allChambers;
}
*/


const std::vector<ME0EtaPartition*>& ME0Geometry::etaPartitions() const{
  return allEtaPartitions;
}

const ME0EtaPartition* ME0Geometry::etaPartition(ME0DetId id) const{
  return dynamic_cast<const ME0EtaPartition*>(idToDetUnit(id));
}


void
ME0Geometry::add(ME0EtaPartition* etaPartition){
  theDets.push_back(etaPartition);
  allEtaPartitions.push_back(etaPartition);
  theEtaPartitions.push_back(etaPartition);
  theEtaPartitionIds.push_back(etaPartition->geographicalId());
  theDetIds.push_back(etaPartition->geographicalId());
  GeomDetType* _t = const_cast<GeomDetType*>(&etaPartition->type());
  theEtaPartitionTypes.push_back(_t);
  theMap.insert(std::pair<DetId,GeomDetUnit*>
		(etaPartition->geographicalId(),etaPartition));
}

