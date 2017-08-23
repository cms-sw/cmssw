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
  // delete all the chambers (which will delete the SL which will delete the
  // layers)
  for (auto & theChamber : theChambers) delete theChamber;
}

const DTGeometry::DetTypeContainer&  DTGeometry::detTypes() const{
  // FIXME - fill it at runtime
  return theDetTypes;
}


void DTGeometry::add(DTChamber* ch) {
  theDets.emplace_back(ch);
  theChambers.emplace_back(ch);
  theMap.insert(DTDetMap::value_type(ch->geographicalId(),ch));
}


void DTGeometry::add(DTSuperLayer* sl) {
  theDets.emplace_back(sl);
  theSuperLayers.emplace_back(sl);
  theMap.insert(DTDetMap::value_type(sl->geographicalId(),sl));
}


void DTGeometry::add(DTLayer* l) {
  theDetUnits.emplace_back(l);
  theDets.emplace_back(l);
  theLayers.emplace_back(l); 
  theMap.insert(DTDetMap::value_type(l->geographicalId(),l));
}


const DTGeometry::DetUnitContainer& DTGeometry::detUnits() const{
  return theDetUnits;
}


const DTGeometry::DetContainer& DTGeometry::dets() const{
  return theDets; 
}


const DTGeometry::DetIdContainer& DTGeometry::detUnitIds() const{
  // FIXME - fill it at runtime
  return theDetUnitIds;
}


const DTGeometry::DetIdContainer& DTGeometry::detIds() const{
  // FIXME - fill it at runtime
  return theDetIds;
}


const GeomDetUnit* DTGeometry::idToDetUnit(DetId id) const{
  return dynamic_cast<const GeomDetUnit*>(idToDet(id));
}


const GeomDet* DTGeometry::idToDet(DetId id) const{
  // Strip away wire#, if any!
  DTLayerId lId(id.rawId());
  DTDetMap::const_iterator i = theMap.find(lId);
  return (i != theMap.end()) ?
    i->second : nullptr ;
}


const std::vector<const DTChamber*>& DTGeometry::chambers() const{
  return theChambers;
}


const std::vector<const DTSuperLayer*>& DTGeometry::superLayers() const{
  return theSuperLayers;
}


const std::vector<const DTLayer*>& DTGeometry::layers() const{
  return theLayers;
}


const DTChamber* DTGeometry::chamber(const DTChamberId& id) const {
  return (const DTChamber*)(idToDet(id));
}


const DTSuperLayer* DTGeometry::superLayer(const DTSuperLayerId& id) const {
  return (const DTSuperLayer*)(idToDet(id));
}


const DTLayer* DTGeometry::layer(const DTLayerId& id) const {
  return (const DTLayer*)(idToDet(id));
}
