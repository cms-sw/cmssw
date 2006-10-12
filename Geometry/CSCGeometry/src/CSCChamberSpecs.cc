#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include <Geometry/CommonDetUnit/interface/GeomDetUnit.h>
#include <Geometry/CommonDetUnit/interface/GeomDetType.h>
#include <Geometry/Surface/interface/TrapezoidalPlaneBounds.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>

using namespace GeomDetEnumerators;

CSCChamberSpecs::CSCChamberSpecs( int iChamberType,
      const TrapezoidalPlaneBounds& bounds,
      const CSCSpecsParcel& fupar,
      const CSCWireGroupPackage& wg )
  : GeomDetType( "CSC", CSC ), theChamberType( iChamberType ), theSpecsValues( fupar )
 {
   LogDebug("CSC") << "Constructing specs for chamber " << 
    theName[iChamberType - 1] << ", type=" << iChamberType << ", this =" << 
     this << "\n";

   // Need to check settings for overall modelling, but only once
   if ( theFirstCall ) {

  //@@ FIXME dependence on calling order!!
  // Expect parameters have been passed from CSCGeometryESModule to set statics here
     whatModelling();

     theFirstCall = false;
   }

  // Most wire info now comes from wire group section of DDD
  //  int nWiresPerGroup = static_cast<int>( specsValue(7) );  
  //  float wireSpacing  = specsValue(9);
  //  float wireAngle    = specsValue(13);

  float wireAngleInDegrees = specsValue(13);
  
  int nstrips        = static_cast<int>( specsValue(5) ) * 2; 
  float stripOffset1 = specsValue(73);
  float stripOffset2 = specsValue(74);

  float phiPitch = this->stripPhiPitch();
  float ctiOffset = 0.;
  if ( useCentreTIOffsets )
    ctiOffset = this->ctiOffset();

  // Build the unique LayerGeometry objects we require for each chamber type.
  // - There are 2 endcaps
  // - Alternate layers of strips are offset w.r.t. each other
  // - In ME11 the wire tilt angle needs to be a constant
  // global value; in the CMS local coordinates this is positive in +z
  // and negative in -z (magnitude 29 degrees as of 2002)

  // Thus we need 4 LGs differing in strip offset and wire angle

  poszOddLayerGeometry = new CSCLayerGeometry( iChamberType, bounds,
     nstrips, -stripOffset1, phiPitch, 
     wg, wireAngleInDegrees, ctiOffset );

  poszEvenLayerGeometry = new CSCLayerGeometry( iChamberType, bounds,
     nstrips, -stripOffset2, phiPitch, 
     wg, wireAngleInDegrees, ctiOffset );

  negzOddLayerGeometry = new CSCLayerGeometry( iChamberType, bounds,
     nstrips, -stripOffset1, phiPitch,
     wg, -wireAngleInDegrees, ctiOffset );

  negzEvenLayerGeometry = new CSCLayerGeometry( iChamberType, bounds,
     nstrips, -stripOffset2, phiPitch,
     wg, -wireAngleInDegrees, ctiOffset );

}


CSCChamberSpecs::~CSCChamberSpecs()
{
  LogDebug("CSC") << "destroying this=" << this << "\n";

  delete poszOddLayerGeometry;
  delete poszEvenLayerGeometry;
  delete negzOddLayerGeometry;
  delete negzEvenLayerGeometry;
}


bool CSCChamberSpecs::operator!=( const CSCChamberSpecs& specs ) const
{
   if ( ( theChamberType != specs.theChamberType ) )
      return true;
    else
      return false;
}

bool CSCChamberSpecs::operator==( const CSCChamberSpecs& specs ) const
{
   return !( operator!=(specs) );
}

const Topology& CSCChamberSpecs::topology() const {
  return *(poszOddLayerGeometry->topology());
}


float CSCChamberSpecs::specsValue( int index ) const {
  return theSpecsValues[ index ];
}

int CSCChamberSpecs::chamberType() const {
  return theChamberType;
}

std::string CSCChamberSpecs::chamberTypeName() const {
  return theName[chamberType() - 1];
}

// Build theChamberType value for a (station, ring) pair
  //             an integer 1-10 corresponding to 
  //        1           for S = 1  and R=A=4 split strips in ME11
  //      2,3,4 =  R+1  for S = 1  and R = 1,2,3
  //      5-10  = 2*S+R for S = 2,3,4 and R = 1,2

int CSCChamberSpecs::whatChamberType( int istation, int iring ) {
   int i = 2 * istation + iring; // i=2S+R
   if ( istation == 1 )
   {
      --i;             // ring 1R -> i=1+R (2S+R-1=1+R for S=1)
      if ( i > 4 )
      {
	  i = 1;       // But ring 1A (R=4) -> i=1
      }
   }   
   return i;
}

CSCChamberSpecs*  CSCChamberSpecs::lookUp( int iChamberType ) {
  CSCChamberSpecs* aSpecs = 0;
  std::map<int, CSCChamberSpecs*, std::less<int> >::const_iterator it = 
                    specsMap.find( iChamberType );
  if (  it != specsMap.end() )       // Requisite Specs already exists
  {
     aSpecs = (*it).second;
  }
  return aSpecs;
}


CSCChamberSpecs* CSCChamberSpecs::build( int iChamberType,
     const std::vector<float>& fpar, const std::vector<float>& fupar, 
     const CSCWireGroupPackage& wg ) {

  // Note arg list order is hbot, htop, apothem, hthickness
  TrapezoidalPlaneBounds bounds( fpar[0], fpar[1], fpar[3], fpar[2] ); 

  CSCChamberSpecs* aSpecs = new CSCChamberSpecs( iChamberType, bounds, fupar, wg );
  specsMap[ iChamberType ] = aSpecs;

  return aSpecs;
}

float CSCChamberSpecs::wireSpacing() const { 
  return poszOddLayerGeometry->wirePitch(); 
}

float CSCChamberSpecs::stripNoise(float timeInterval) const {
  const float pF_cm = 0.75;
  return (constantNoise() + e_pF() * pF_cm * poszOddLayerGeometry->length() )
         / sqrt(timeInterval/100.);
}

float CSCChamberSpecs::gasGain() const {
  // ME1/1 makes up for its small gap by a big electronics gain
  return 300000.;
}

float CSCChamberSpecs::chargePerCount() const {
  if(theChamberType <= 2) {
    return 0.25;
  } else {
    return 0.5;
  }
}

void CSCChamberSpecs::whatModelling() {
  // Static function to dump user-selected overall modelling parameters
  // This will be called just _once_ in a job!

  LogDebug("CSC") << "setModelling entered... " << "\n";

  std::string sm = " ";
  if ( useRadialStrips ) 
    sm = "RADIAL";
  else
    sm = "TRAPEZOIDAL";

  edm::LogInfo("CSC") << myName << ": strips are modelled as " << sm << "\n";

  std::string gs = " ";
  if ( gangedstripsME1a ) 
    gs = "GANGED";
  else
    gs = "UNGANGED";

  edm::LogInfo("CSC") << myName << ": in ME1a use " << gs << " strips" << "\n";

  std::string wo = " ";
  if ( onlywiresME1a ) 
    wo = "WIRES ONLY";
  else
    wo = "WIRES & STRIPS";

  edm::LogInfo("CSC") << myName << ": in ME1a use  " << wo << "\n";

  std::string wg = " ";
  if ( useRealWireGeometry ) 
    wg = "REAL";
  else
    wg = "PSEUDO";

  edm::LogInfo("CSC") << myName << ": wires are modelled using " << wg << " wire geometry " << "\n";

  std::string cti = " ";
  if ( useCentreTIOffsets )
    cti = "WITH";
  else
    cti = "WITHOUT";

  edm::LogInfo("CSC") << myName << ": strips are placed relative to the gas volume " << cti << " offsets " << "\n";
}

// Define the specsMap 
std::map<int, CSCChamberSpecs*, std::less<int> > CSCChamberSpecs::specsMap;

// Define the phi width of strips in each chamber type
const float CSCChamberSpecs::stripDeltaPhi[] =
{ 2.96, 2.96, 2.33, 2.16, 4.65, 2.33, 4.65, 2.33, 4.65, 2.33 };

// Define the hacked-in offsets for the whereStripsMeet calculation
// @@ ME1/3 does not get adjusted, and no info for ME4/2 
const float CSCChamberSpecs::centreToIntersectionOffset[] = 
{ 16.033752, 16.033752, 8.694855, 0., -0.576782, 5.048706, -0.693268, 5.048706, 5.855377, 0. };

// Define the name of each chamber type
const std::string CSCChamberSpecs::theName[] =
  {"ME1/a", "ME1/b", "ME1/2", "ME1/3", "ME2/1", "ME2/2", "ME3/1", "ME3/2",
   "ME4/1", "ME4/2" };

// Define name of this class
const std::string CSCChamberSpecs::myName = "CSCChamberSpecs";
 
// Define the static bools
bool CSCChamberSpecs::theFirstCall = true;
bool CSCChamberSpecs::gangedstripsME1a = true;
bool CSCChamberSpecs::onlywiresME1a = false;
bool CSCChamberSpecs::useRadialStrips = true;
bool CSCChamberSpecs::useRealWireGeometry = false; // pseudo wire geometry
bool CSCChamberSpecs::useCentreTIOffsets = false;
