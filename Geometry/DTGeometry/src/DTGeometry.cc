/** \file
 *
 *  $Date: 2006/02/22 11:06:45 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - CERN
 */

#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <Geometry/CommonDetUnit/interface/GeomDetUnit.h>
#include <Geometry/CommonDetUnit/interface/GeomDetType.h>

#include <algorithm>
#include <iostream>

DTGeometry::DTGeometry() {}

DTGeometry::~DTGeometry(){
  // delete all the chambers (which will delete the SL which will delete the
  // layers)
  for (std::vector<DTChamber*>::const_iterator ich=theChambers.begin();
       ich!=theChambers.end(); ++ich) delete (*ich);
}

const DTGeometry::DetTypeContainer&  DTGeometry::detTypes() const{
  // FIXME
  static DetTypeContainer result;
  return result;
}


void DTGeometry::add(DTChamber* ch) {
  theChambers.push_back(ch);
  theChambersMap.insert(std::pair<DetId,DTChamber*>(ch->geographicalId(),ch));
}


void DTGeometry::add(DTSuperLayer* sl) {
  theSuperLayers.push_back(sl);
  theSuperLayersMap.insert(std::pair<DetId,DTSuperLayer*>(sl->geographicalId(),sl));
}


void DTGeometry::add(DTLayer* l) {
  theLayers.push_back(l); 
  theLayersMap.insert(std::pair<DetId,DTLayer*>(l->geographicalId(),l));
  theDetUnits.push_back(l);
}


const DTGeometry::DetUnitContainer& DTGeometry::detUnits() const{
  return theDetUnits;
}


const DTGeometry::DetContainer& DTGeometry::dets() const{
  return theDets; // FIXME - empty!
}


const DTGeometry::DetIdContainer& DTGeometry::detUnitIds() const{
  // FIXME
  static DetIdContainer result;
  return result;
}


const DTGeometry::DetIdContainer& DTGeometry::detIds() const{
  // FIXME
  static DetIdContainer result;
  return result;
}


const GeomDetUnit* DTGeometry::idToDetUnit(DetId id) const{
  return (theLayersMap.find(id) != theLayersMap.end()) ?
    (theLayersMap.find(id))->second : 0 ;
}


const GeomDet* DTGeometry::idToDet(DetId id) const{
  // FIXME!!! should search in all GeomDet*!!!
  return idToDetUnit(id);
}


const std::vector<DTChamber*>& DTGeometry::chambers() const{
  return theChambers;
}


const std::vector<DTSuperLayer*>& DTGeometry::superLayers() const{
  return theSuperLayers;
}


const std::vector<DTLayer*>& DTGeometry::layers() const{
  return theLayers;
}


const DTChamber* DTGeometry::chamber(const DTChamberId& id) const {
  return (theChambersMap.find(id) != theChambersMap.end()) ?
    (theChambersMap.find(id))->second : 0 ;
}


const DTSuperLayer* DTGeometry::superLayer(const DTSuperLayerId& id) const {
  return (theSuperLayersMap.find(id) != theSuperLayersMap.end()) ?
    (theSuperLayersMap.find(id))->second : 0 ;
}


const DTLayer* DTGeometry::layer(const DTLayerId& id) const {
  return (theLayersMap.find(id) != theLayersMap.end()) ?
    (theLayersMap.find(id))->second : 0 ;
}
