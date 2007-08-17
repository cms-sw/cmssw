#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"
#include "Geometry/CSCGeometry/src/CSCWireGroupPackage.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

#include <string>

CSCGeometry::CSCGeometry():  debugV(false), gangedstripsME1a(true), 
   onlywiresME1a(false), useRealWireGeometry(true), useCentreTIOffsets(false) {
   if ( debugV ) queryModelling();
}

CSCGeometry::~CSCGeometry(){

  // delete all the chambers (which will delete the layers)
  for (ChamberContainer::const_iterator ich=theChambers.begin();
       ich!=theChambers.end(); ++ich) delete (*ich);

  // delete specs
  for ( std::map<int, CSCChamberSpecs*, std::less<int> >::const_iterator it =
	   specsMap.begin(); it!=specsMap.end(); ++it) {
    delete (*it).second; // they are never shared per chamber type so should be no possible double deletion.
  }

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

void CSCGeometry::queryModelling() const {
  // Dump user-selected overall modelling parameters.
  // Only requires calling once per job.

  LogTrace("CSCGeometry|CSC")  << "CSCGeometry::queryModelling entered...";

  std::string gs = " ";
  if ( gangedstripsME1a )
    gs = "GANGED";
  else
    gs = "UNGANGED";

  edm::LogInfo("CSC") << "CSCGeometry: in ME1a use " << gs << " strips" << "\n";

  std::string wo = " ";
  if ( onlywiresME1a )
    wo = "WIRES ONLY";
  else
    wo = "WIRES & STRIPS";

  edm::LogInfo("CSC") << "CSCGeometry: in ME1a use  " << wo << "\n";

  std::string wg = " ";
  if ( useRealWireGeometry )
    wg = "REAL";
  else
    wg = "PSEUDO";

  edm::LogInfo("CSC") << "CSCGeometry: wires are modelled using " << wg << " wire geometry " << "\n";

  std::string cti = " ";
  if ( useCentreTIOffsets )
    cti = "WITH";
  else
    cti = "WITHOUT";

  edm::LogInfo("CSC") << "CSCGeometry: strip plane centre-to-intersection ideal " << cti << " corrections " << "\n";
}

CSCChamberSpecs* CSCGeometry::findSpecs( int iChamberType ) {
  CSCChamberSpecs* aSpecs = 0;
  std::map<int, CSCChamberSpecs*, std::less<int> >::const_iterator it =
    specsMap.find( iChamberType );
  if (  it != specsMap.end() )       // Requisite Specs already exists
    {
      aSpecs = (*it).second;
    }
  return aSpecs;
} 

CSCChamberSpecs* CSCGeometry::buildSpecs( int iChamberType,
					 const std::vector<float>& fpar,
					 const std::vector<float>& fupar,
					 const CSCWireGroupPackage& wg ) {

  // Note arg list order is hbot, htop, apothem, hthickness
  TrapezoidalPlaneBounds bounds( fpar[0], fpar[1], fpar[3], fpar[2] );

  CSCChamberSpecs* aSpecs = new CSCChamberSpecs( this, iChamberType, bounds, fupar, wg );
  specsMap[ iChamberType ] = aSpecs;

  return aSpecs;
}
