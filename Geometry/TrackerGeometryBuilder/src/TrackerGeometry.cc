#include <typeinfo>

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <algorithm>
#include <iostream>
#include <map>

TrackerGeometry::TrackerGeometry(GeometricDet const* gd) :  theTrackerDet(gd){}

TrackerGeometry::~TrackerGeometry() {
    for (auto d : theDets) delete const_cast<GeomDet*>(d);
    for (auto d : theDetTypes) delete const_cast<GeomDetType*>(d);
}

void TrackerGeometry::finalize() {
    theDetTypes.shrink_to_fit();  // owns the DetTypes
    theDetUnits.shrink_to_fit();  // they're all also into 'theDets', so we assume 'theDets' owns them
    theDets.shrink_to_fit();     // owns *ONLY* the GeomDet * corresponding to GluedDets.
    theDetUnitIds.shrink_to_fit();
    theDetIds.shrink_to_fit();
  
    thePXBDets.shrink_to_fit(); // not owned: they're also in 'theDets'
    thePXFDets.shrink_to_fit(); // not owned: they're also in 'theDets'
    theTIBDets.shrink_to_fit(); // not owned: they're also in 'theDets'
    theTIDDets.shrink_to_fit(); // not owned: they're also in 'theDets'
    theTOBDets.shrink_to_fit(); // not owned: they're also in 'theDets'
    theTECDets.shrink_to_fit(); // not owned: they're also in 'theDets'
}


void TrackerGeometry::addType(GeomDetType const * p) {
  theDetTypes.push_back(p);  // add to vector
}

void TrackerGeometry::addDetUnit(GeomDetUnit const * p) {
  // set index
  const_cast<GeomDetUnit *>(p)->setIndex(theDetUnits.size());
  theDetUnits.push_back(p);  // add to vector
  theMapUnit.insert(std::make_pair(p->geographicalId().rawId(),p));
}

void TrackerGeometry::addDetUnitId(DetId p){
  theDetUnitIds.push_back(p);
}

void TrackerGeometry::addDet(GeomDet const * p) {
  theDets.push_back(p);  // add to vector
  theMap.insert(std::make_pair(p->geographicalId().rawId(),p));
  DetId id(p->geographicalId());
  switch(id.subdetId()){
  case PixelSubdetector::PixelBarrel:
    thePXBDets.push_back(p);
    break;
  case PixelSubdetector::PixelEndcap:
    thePXFDets.push_back(p);
    break;
  case StripSubdetector::TIB:
    theTIBDets.push_back(p);
    break;
  case StripSubdetector::TID:
    theTIDDets.push_back(p);
    break;
  case StripSubdetector::TOB:
    theTOBDets.push_back(p);
    break;
  case StripSubdetector::TEC:
    theTECDets.push_back(p);
    break;
  default:
    edm::LogError("TrackerGeometry")<<"ERROR - I was expecting a Tracker Subdetector, I got a "<<id.subdetId();
  }


}

void TrackerGeometry::addDetId(DetId p){
  theDetIds.push_back(p);
}

const TrackerGeometry::DetUnitContainer&
TrackerGeometry::detUnits() const
{
  return theDetUnits;
}

const TrackerGeometry::DetContainer&
TrackerGeometry::dets() const
{
  return theDets;
}

const TrackerGeometry::DetContainer&
TrackerGeometry::detsPXB() const
{
  return thePXBDets;
}

const TrackerGeometry::DetContainer&
TrackerGeometry::detsPXF() const
{
  return thePXFDets;
}

const TrackerGeometry::DetContainer&
TrackerGeometry::detsTIB() const
{
  return theTIBDets;
}

const TrackerGeometry::DetContainer&
TrackerGeometry::detsTID() const
{
  return theTIDDets;
}

const TrackerGeometry::DetContainer&
TrackerGeometry::detsTOB() const
{
  return theTOBDets;
}

const TrackerGeometry::DetContainer&
TrackerGeometry::detsTEC() const
{
  return theTECDets;
}

const GeomDetUnit* 
TrackerGeometry::idToDetUnit(DetId s)const
{
  mapIdToDetUnit::const_iterator p=theMapUnit.find(s.rawId());
  if (p != theMapUnit.end())
    return (p)->second;
  edm::LogError("TrackerGeometry")<<"Invalid DetID: no GeomDetUnit associated";
  GeomDetUnit* geom = 0;
  return geom;
}

const GeomDet* 
TrackerGeometry::idToDet(DetId s)const
{
  mapIdToDet::const_iterator p=theMap.find(s.rawId());
  if (p != theMap.end())
    return (p)->second;
  edm::LogError("TrackerGeometry")<<"Invalid DetID: no GeomDet associated";
  GeomDet* geom = 0;
  return geom;
}

const TrackerGeometry::DetTypeContainer&  
TrackerGeometry::detTypes()   const 
{
  return theDetTypes;
}


const TrackerGeometry::DetIdContainer&  
TrackerGeometry::detUnitIds()   const 
{
  return theDetUnitIds;
}

const TrackerGeometry::DetIdContainer&  
TrackerGeometry::detIds()   const 
{
  return theDetIds;
}


