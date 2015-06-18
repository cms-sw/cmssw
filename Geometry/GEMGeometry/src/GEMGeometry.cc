/** Implementation of the Model for GEM Geometry
 *
 *  \author M. Maggi - INFN Bari
 */

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

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
  return (i != theMap.end()) ? i->second : 0;
}

const std::vector<const GEMRegion*>& GEMGeometry::regions() const {
  return allRegions;
}

const std::vector<const GEMStation*>& GEMGeometry::stations() const {
  return allStations;
}

const std::vector<const GEMRing*>& GEMGeometry::rings() const{
  return allRings;
}

const std::vector<const GEMSuperChamber*>& GEMGeometry::superChambers() const {
  return allSuperChambers;
}

const std::vector<const GEMChamber*>& GEMGeometry::chambers() const {
  return allChambers;
}

const std::vector<const GEMEtaPartition*>& GEMGeometry::etaPartitions() const{
  return allEtaPartitions;
}

const GEMRegion* GEMGeometry::region(int re) const{
  for (auto region : allRegions) {
    if (re != region->region()) continue;
    return region;
  }
  return 0;
}

const GEMStation* GEMGeometry::station(int re, int st) const{ 
  for (auto station : allStations) {
    if (re != station->region() || st != station->station()) continue;
    return station;
  }
  return 0;
}

const GEMRing* GEMGeometry::ring(int re, int st, int ri) const{
  for (auto ring : allRings) {
    if (re != ring->region() || st != ring->station() || ri != ring->ring()) continue;	
    return ring;
  }
  return 0;
}

const GEMSuperChamber* GEMGeometry::superChamber(GEMDetId id) const{
  for (auto sch : allSuperChambers){
    if (sch->id() != id) continue;
    return sch;
  }
  return 0;
}

const GEMChamber* GEMGeometry::chamber(GEMDetId id) const{ 
  return dynamic_cast<const GEMChamber*>(idToDet(id.chamberId()));
}

const GEMEtaPartition* GEMGeometry::etaPartition(GEMDetId id) const{
  return dynamic_cast<const GEMEtaPartition*>(idToDetUnit(id));
}

void
GEMGeometry::add(GEMRegion* region){
  allRegions.push_back(region);
}

void
GEMGeometry::add(GEMStation* station){
  allStations.push_back(station);
}

void
GEMGeometry::add(GEMRing* ring){
  allRings.push_back(ring);
}

void
GEMGeometry::add(GEMSuperChamber* superChamber){
  allSuperChambers.push_back(superChamber);
  theDets.push_back(superChamber);
  theDetIds.push_back(superChamber->geographicalId());
  theMap.insert(std::pair<DetId,GeomDet*>
  		(superChamber->geographicalId(),superChamber));
}

void
GEMGeometry::add(GEMEtaPartition* etaPartition){
  theDets.push_back(etaPartition);
  allEtaPartitions.push_back(etaPartition);
  theEtaPartitions.push_back(etaPartition);
  theEtaPartitionIds.push_back(etaPartition->geographicalId());
  theDetIds.push_back(etaPartition->geographicalId());
  theEtaPartitionTypes.push_back(&etaPartition->type());
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
