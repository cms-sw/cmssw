#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

CSCGeometry::CSCGeometry(){}


CSCGeometry::~CSCGeometry(){

}  

void CSCGeometry::addChamber( CSCDetId id, Pointer2Chamber chamber ){
  theSystemOfChambers.insert( MapId2Chamber::value_type( id, chamber ) );
}

Pointer2Chamber CSCGeometry::getChamber( CSCDetId id ) const {
  MapId2Chamber::const_iterator it = theSystemOfChambers.find( id );
  if ( it != theSystemOfChambers.end() ) {
    return (*it).second;
  }
  else {
    return Pointer2Chamber();
  }
}

void CSCGeometry::addDet(GeomDetUnit* p) {
  theDets.push_back(p);  // add to vector
  theMap.insert(std::pair<DetId,GeomDetUnit*>(p->geographicalId(),p));
}

void CSCGeometry::addDetType(GeomDetType* p) {
  theDetTypes.push_back(p);  // add to vector
}

void CSCGeometry::addDetId(DetId p){
  theDetIds.push_back(p);
}

const CSCGeometry::DetContainer& CSCGeometry::dets() const
{
  return theDets;
}

const GeomDetUnit* CSCGeometry::idToDet(DetId s) const
{
  if (theMap.find(s) != theMap.end()) {
    return (theMap.find(s))->second;
  }
  else {
    return 0;
  }
}

const CSCGeometry::DetTypeContainer& CSCGeometry::detTypes() const 
{
  return theDetTypes;
}

const CSCGeometry::DetIdContainer& CSCGeometry::detIds() const 
{
  return theDetIds;
}

const ChamberContainer CSCGeometry::chambers() const
{
  ChamberContainer chc;
  for( MapId2Chamber::const_iterator it = theSystemOfChambers.begin();
       it != theSystemOfChambers.end(); ++it ) {
    chc.push_back( (*it).second.get() );
  }
  return chc;
}

const LayerContainer CSCGeometry::layers() const
{
  LayerContainer lc;
  for( DetContainer::const_iterator it = theDets.begin();
       it != theDets.end(); ++it ) {
    CSCLayer* layer = dynamic_cast<CSCLayer*>( *it );
    if ( layer ) lc.push_back( layer );
  }
  return lc;
}


