/** \file
 *
 *  \author N. Amapane - CERN
 */

#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>

#include <algorithm>
#include <iostream>

DTGeometry::DTGeometry() {}

DTGeometry::~DTGeometry(){
}

const DTGeometry::DetTypeContainer&  DTGeometry::detTypes() const {
  // FIXME - fill it at runtime
  return theDetTypes;
}

void DTGeometry::add( std::shared_ptr< DTChamber > ch ) {
  theDets.emplace_back(ch);
  theChambers.emplace_back(ch);
  theMap.insert(DTDetMap::value_type(ch->geographicalId(),ch));
}

void DTGeometry::add( std::shared_ptr< DTSuperLayer > sl ) {
  theDets.emplace_back(sl);
  theSuperLayers.emplace_back(sl);
  theMap.insert(DTDetMap::value_type(sl->geographicalId(),sl));
}

void DTGeometry::add( std::shared_ptr< DTLayer > l ) {
  theDetUnits.emplace_back(l);
  theDets.emplace_back(l);
  theLayers.emplace_back(l); 
  theMap.insert(DTDetMap::value_type(l->geographicalId(),l));
}

const DTGeometry::DetContainer& DTGeometry::detUnits() const {
  return theDetUnits;
}

const DTGeometry::DetContainer& DTGeometry::dets() const {
  return theDets; 
}

const DTGeometry::DetIdContainer& DTGeometry::detUnitIds() const {
  // FIXME - fill it at runtime
  return theDetUnitIds;
}

const DTGeometry::DetIdContainer& DTGeometry::detIds() const {
  // FIXME - fill it at runtime
  return theDetIds;
}

const std::shared_ptr< GeomDet >
DTGeometry::idToDetUnit( DetId id ) const {
  return std::static_pointer_cast< GeomDet >(idToDet(id));
}

const std::shared_ptr< GeomDet >
DTGeometry::idToDet(DetId id) const {
  // Strip away wire#, if any!
  DTLayerId lId(id.rawId());
  DTDetMap::const_iterator i = theMap.find(lId);
  return (i != theMap.end()) ?
    i->second : nullptr ;
}

const std::vector< std::shared_ptr< DTChamber >>&
DTGeometry::chambers() const {
  return theChambers;
}

const std::vector< std::shared_ptr< DTSuperLayer >>&
DTGeometry::superLayers() const {
  return theSuperLayers;
}

const std::vector< std::shared_ptr< DTLayer >>&
DTGeometry::layers() const {
  return theLayers;
}

const std::shared_ptr< DTChamber >
DTGeometry::chamber( const DTChamberId& id ) const {
  return std::static_pointer_cast< DTChamber >( idToDet( id ));
}

const std::shared_ptr< DTSuperLayer >
DTGeometry::superLayer( const DTSuperLayerId& id ) const {
  return std::static_pointer_cast< DTSuperLayer >( idToDet( id ));
}

const std::shared_ptr< DTLayer >
DTGeometry::layer(const DTLayerId& id) const {
  return std::static_pointer_cast< DTLayer >( idToDet( id ));
}
