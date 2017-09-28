/** Implementation of the Model for ME0 Geometry
 *
 *  \author M. Maggi - INFN Bari
 */

#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

ME0Geometry::ME0Geometry(){}
ME0Geometry::~ME0Geometry(){}  

const ME0Geometry::DetTypeContainer&  ME0Geometry::detTypes() const{
  return theEtaPartitionTypes;
}

const ME0Geometry::DetContainer& ME0Geometry::detUnits() const{
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

const std::shared_ptr< GeomDet >
ME0Geometry::idToDetUnit( DetId id ) const {
  return idToDet( id );
}

const std::shared_ptr< GeomDet >
ME0Geometry::idToDet( DetId id ) const {
  mapIdToDet::const_iterator i = theMap.find( id );
  return (i != theMap.end()) ?
    i->second : nullptr ;
}

const std::vector< std::shared_ptr< ME0Chamber >>&
ME0Geometry::chambers() const {
  return allChambers;
}

const std::vector< std::shared_ptr< ME0Layer >>&
ME0Geometry::layers() const {
  return allLayers;
}

const std::vector< std::shared_ptr< ME0EtaPartition >>&
ME0Geometry::etaPartitions() const {
  return allEtaPartitions;
}

const std::shared_ptr< ME0EtaPartition >
ME0Geometry::etaPartition( ME0DetId id ) const{
  return std::static_pointer_cast< ME0EtaPartition >( idToDetUnit( id ));
}

const std::shared_ptr< ME0Layer >
ME0Geometry::layer( ME0DetId id ) const {
  return std::static_pointer_cast< ME0Layer >( idToDetUnit( id.layerId()));
}

const std::shared_ptr< ME0Chamber >
ME0Geometry::chamber( ME0DetId id ) const {
  return std::static_pointer_cast< ME0Chamber >( idToDetUnit( id.chamberId()));
}

void
ME0Geometry::add( std::shared_ptr< ME0EtaPartition > etaPartition ) {
  allEtaPartitions.emplace_back(etaPartition);
  theEtaPartitions.emplace_back(etaPartition);
  theEtaPartitionIds.emplace_back(etaPartition->geographicalId());
  theDets.emplace_back(etaPartition);
  theDetIds.emplace_back(etaPartition->geographicalId());
  theEtaPartitionTypes.emplace_back(&etaPartition->type());
  theMap.insert( std::pair< DetId, std::shared_ptr< ME0EtaPartition > >
		 (etaPartition->geographicalId(),etaPartition));
}

void
ME0Geometry::add( std::shared_ptr< ME0Layer > layer ) {
  allLayers.emplace_back(layer);
  theDets.emplace_back(layer);
  theDetIds.emplace_back(layer->geographicalId());
  theEtaPartitionTypes.emplace_back(&layer->type());
  theMap.insert( std::pair< DetId, std::shared_ptr< GeomDet > >
		 (layer->geographicalId(),layer));
}

void
ME0Geometry::add( std::shared_ptr< ME0Chamber > chamber ) {
  allChambers.emplace_back(chamber);
  theDets.emplace_back(chamber);
  theDetIds.emplace_back(chamber->geographicalId());
  theMap.insert( std::pair< DetId, std::shared_ptr< GeomDet > >
		 (chamber->geographicalId(),chamber));
}
