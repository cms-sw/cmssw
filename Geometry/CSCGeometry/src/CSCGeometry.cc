#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"


CSCGeometry::CSCGeometry(){}


CSCGeometry::~CSCGeometry(){
  // delete all the chambers (which will delete the SL which will delete the
  // layers)
  for (ChamberContainer::const_iterator ich=theChambers.begin();
       ich!=theChambers.end(); ++ich) delete (*ich);
}  


void CSCGeometry::addChamber(CSCChamber* ch){
  theChambers.push_back(ch);
  addDet(ch);
}


void CSCGeometry::addLayer(CSCLayer* l) {
  theDetUnits.push_back(l);
  theLayers.push_back(l);
  theDetTypes.push_back(const_cast<GeomDetType*>(&(l->type()))); //@@ FIXME drop const_cast asap!
  theDetUnitIds.push_back(l->geographicalId());
  addDet(l);
}


void CSCGeometry::addDetType(GeomDetType* type) {
  theDetTypes.push_back(type);
}


void CSCGeometry::addDet(GeomDet* det){
  theDets.push_back(det);  
  theDetIds.push_back(det->geographicalId());
  theMap.insert(CSCDetMap::value_type(det->geographicalId(),det));
}


const CSCGeometry::DetTypeContainer& CSCGeometry::detTypes() const 
{
  return theDetTypes;
}


const CSCGeometry::DetUnitContainer& CSCGeometry::detUnits() const
{
  return theDetUnits;
}


const CSCGeometry::DetContainer& CSCGeometry::dets() const
{
  return theDets;
}


const CSCGeometry::DetIdContainer& CSCGeometry::detUnitIds() const 
{
  return theDetUnitIds;
}


const CSCGeometry::DetIdContainer& CSCGeometry::detIds() const 
{
  return theDetIds;
}


const GeomDetUnit* CSCGeometry::idToDetUnit(DetId id) const
{
  return dynamic_cast<const GeomDetUnit*>(idToDet(id));
}


const GeomDet* CSCGeometry::idToDet(DetId id) const{
  CSCDetMap::const_iterator i = theMap.find(id);
  return (i != theMap.end()) ?
    i->second : 0 ;
}


const CSCGeometry::ChamberContainer& CSCGeometry::chambers() const
{
  return theChambers;
}


const CSCGeometry::LayerContainer& CSCGeometry::layers() const
{
//   for( DetUnitContainer::const_iterator it = theDetUnits.begin();
//        it != theDetUnits.end(); ++it ) {
//     CSCLayer* layer = dynamic_cast<CSCLayer*>( *it );
//     if ( layer ) theLayers.push_back( layer );
//   }
  return theLayers;
}


const CSCChamber* CSCGeometry::chamber(CSCDetId id) const {
  CSCDetId id1(id.endcap(), id.station(), id.ring(), id.chamber(), 0);
  return dynamic_cast<const CSCChamber*>(idToDet(id1));
}


const CSCLayer* CSCGeometry::layer(CSCDetId id) const {
  return dynamic_cast<const CSCLayer*>(idToDetUnit(id));
}
