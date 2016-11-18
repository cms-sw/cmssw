/** Implementation of the Model for ME0 Geometry
 *
 *  \author M. Maggi - INFN Bari
 */

#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

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


const std::vector<ME0Chamber const*>& ME0Geometry::chambers() const {
  return allChambers;
}
 

const std::vector<ME0Layer const*>& ME0Geometry::layers() const {
  return allLayers;
}


const std::vector<ME0EtaPartition const*>& ME0Geometry::etaPartitions() const{
  return allEtaPartitions;
}


const ME0EtaPartition* ME0Geometry::etaPartition(ME0DetId id) const{
  return dynamic_cast<const ME0EtaPartition*>(idToDetUnit(id));
}


const ME0Layer* ME0Geometry::layer(ME0DetId id) const{
  return dynamic_cast<const ME0Layer*>(idToDetUnit(id.layerId()));
}

const ME0Chamber* ME0Geometry::chamber(ME0DetId id) const{
  return dynamic_cast<const ME0Chamber*>(idToDetUnit(id.chamberId()));
}


void
ME0Geometry::add(ME0EtaPartition* etaPartition){
  allEtaPartitions.push_back(etaPartition);
  theEtaPartitions.push_back(etaPartition);
  theEtaPartitionIds.push_back(etaPartition->geographicalId());
  theDets.push_back(etaPartition);
  theDetIds.push_back(etaPartition->geographicalId());
  theEtaPartitionTypes.push_back(&etaPartition->type());
  theMap.insert(std::pair<DetId,GeomDetUnit*>
		(etaPartition->geographicalId(),etaPartition));
}


void
ME0Geometry::add(ME0Layer* layer){
  allLayers.push_back(layer);
  // theLayers.push_back(layer);                      ??? what would this be fore?
  // theLayerIds.push_back(layer->geographicalId());  ??? what would this be fore?
  theDets.push_back(layer);
  theDetIds.push_back(layer->geographicalId());
  theEtaPartitionTypes.push_back(&layer->type());
  theMap.insert(std::pair<DetId,GeomDetUnit*>
		(layer->geographicalId(),layer));
}


void
ME0Geometry::add(ME0Chamber* chamber){
  allChambers.push_back(chamber);
  theDets.push_back(chamber);
  theDetIds.push_back(chamber->geographicalId());
  theMap.insert(std::pair<DetId,GeomDet*>
		(chamber->geographicalId(),chamber));
}

