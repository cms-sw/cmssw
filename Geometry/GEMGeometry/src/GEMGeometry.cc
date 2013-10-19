/** Implementation of the Model for GEM Geometry
 *
 *  \author M. Maggi - INFN Bari
 */

#include <Geometry/GEMGeometry/interface/GEMGeometry.h>
#include <Geometry/CommonDetUnit/interface/GeomDetUnit.h>

GEMGeometry::GEMGeometry(){}


GEMGeometry::~GEMGeometry(){}  

 
const GEMGeometry::DetTypeContainer&  GEMGeometry::detTypes() const{
  return theEtaPartitionTypes;
}


const GEMGeometry::DetUnitContainer& GEMGeometry::detUnits() const{
  return theEtaPartitions;
}


const GEMGeometry::DetContainer& GEMGeometry::dets() const{
  return theDets;
}

  
const GEMGeometry::DetIdContainer& GEMGeometry::detUnitIds() const{
  return theEtaPartitionIds;
}


const GEMGeometry::DetIdContainer& GEMGeometry::detIds() const{
  return theDetIds;
}


const GeomDetUnit* GEMGeometry::idToDetUnit(DetId id) const{
  return dynamic_cast<const GeomDetUnit*>(idToDet(id));
}


const GeomDet* GEMGeometry::idToDet(DetId id) const{
  mapIdToDet::const_iterator i = theMap.find(id);
  return (i != theMap.end()) ? i->second : 0 ;
}


/*
const std::vector<GEMSuperChamber*>& GEMGeometry::superChambers() const {
  return allSuperChambers;
}
*/

const std::vector<GEMChamber*>& GEMGeometry::chambers() const {
  return allChambers;
}


const std::vector<const GEMEtaPartition*>& GEMGeometry::etaPartitions() const{
  return allEtaPartitions;
}


/*
FIXME
const GEMSuperChamber* GEMGeometry::superChamber(GEMDetId id) const{
  GEMDetId schId = GEMDetId(id.region(),id.ring(),id.station(),0,id.chamber(),0);
  return dynamic_cast<const GEMSuperChamber*>(idToDet(schId));
}
*/

const GEMChamber* GEMGeometry::chamber(GEMDetId id) const{ 
  return dynamic_cast<const GEMChamber*>(idToDet(id.chamberId()));
}


const GEMEtaPartition* GEMGeometry::etaPartition(GEMDetId id) const{
  return dynamic_cast<const GEMEtaPartition*>(idToDetUnit(id));
}


void
GEMGeometry::add(GEMEtaPartition* etaPartition){
  theDets.push_back(etaPartition);
  allEtaPartitions.push_back(etaPartition);
  theEtaPartitions.push_back(etaPartition);
  theEtaPartitionIds.push_back(etaPartition->geographicalId());
  theDetIds.push_back(etaPartition->geographicalId());
  GeomDetType const* _t = &etaPartition->type();
  theEtaPartitionTypes.push_back(_t);
  theMap.insert(std::pair<DetId,const GeomDetUnit*>
		(etaPartition->geographicalId(),etaPartition));
}


void
GEMGeometry::add(GEMChamber* chamber){
  allChambers.push_back(chamber);
  theDets.push_back(chamber);
  theDetIds.push_back(chamber->geographicalId());
  theMap.insert(std::pair<DetId,GeomDet*>
		(chamber->geographicalId(),chamber));
}

/*
FIXME
void
GEMGeometry::add(GEMSuperChamber* superChamber){
  allSuperChambers.push_back(superChamber);
  theDets.push_back(superChamber);
  theDetIds.push_back(superChamber->geographicalId());
  theMap.insert(std::pair<DetId,GeomDet*>
 		(superChamber->geographicalId(),superChamber));
}
*/
