#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

#include <string>

CSCGeometry::CSCGeometry():  debugV_(false), gangedstripsME1a_(true), 
   onlywiresME1a_(false), realWireGeometry_(true), useCentreTIOffsets_(false) {
   if ( debugV_ ) queryModelling();
}

CSCGeometry::CSCGeometry( bool dbgv, bool gangedstripsME1a, bool onlywiresME1a, bool realWireGeometry, bool useCentreTIOffsets ) :  
   debugV_(dbgv), gangedstripsME1a_( gangedstripsME1a ), onlywiresME1a_( onlywiresME1a ), 
   realWireGeometry_( realWireGeometry ), useCentreTIOffsets_( useCentreTIOffsets ) {
   if ( debugV_ ) queryModelling();
}

CSCGeometry::~CSCGeometry(){

  // delete all the chambers (which will delete the layers)
  for (ChamberContainer::const_iterator ich=theChambers.begin();
       ich!=theChambers.end(); ++ich) delete (*ich);

  // delete specs
  for ( CSCSpecsContainer::const_iterator it =
	   specsContainer.begin(); it!=specsContainer.end(); ++it) {
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

  edm::LogInfo("CSC") << "CSCGeometry version 18-Oct-2012 queryModelling...\n";

  std::string gs = " ";
  if ( gangedstripsME1a_ )
    gs = "GANGED";
  else
    gs = "UNGANGED";

  edm::LogInfo("CSC") << "CSCGeometry: in ME1a use " << gs << " strips" << "\n";

  std::string wo = " ";
  if ( onlywiresME1a_ )
    wo = "WIRES ONLY";
  else
    wo = "WIRES & STRIPS";

  edm::LogInfo("CSC") << "CSCGeometry: in ME1a use  " << wo << "\n";

  std::string wg = " ";
  if ( realWireGeometry_ )
    wg = "REAL";
  else
    wg = "PSEUDO";

  edm::LogInfo("CSC") << "CSCGeometry: wires are modelled using " << wg << " wire geometry " << "\n";

  std::string cti = " ";
  if ( useCentreTIOffsets_ )
    cti = "WITH";
  else
    cti = "WITHOUT";

  edm::LogInfo("CSC") << "CSCGeometry: strip plane centre-to-intersection ideal " << cti << " corrections " << "\n";
}

const CSCChamberSpecs* CSCGeometry::findSpecs( int iChamberType ) {
  const CSCChamberSpecs* aSpecs = 0;
  CSCSpecsContainer::const_iterator it = specsContainer.find( iChamberType );
  if (  it != specsContainer.end() )  aSpecs = (*it).second;
  return aSpecs;
} 

const CSCChamberSpecs* CSCGeometry::buildSpecs( int iChamberType,
					 const std::vector<float>& fpar,
					 const std::vector<float>& fupar,
					 const CSCWireGroupPackage& wg ) {

  // Note arg list order is hbot, htop, apothem, hthickness
  TrapezoidalPlaneBounds bounds( fpar[0], fpar[1], fpar[3], fpar[2] );
  const CSCChamberSpecs* aSpecs = new CSCChamberSpecs( this, iChamberType, bounds, fupar, wg );
  specsContainer[ iChamberType ] = aSpecs;
  return aSpecs;
}
