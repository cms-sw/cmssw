#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include <Geometry/CSCGeometry/src/CSCWireGroupPackage.h>
#include <DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

using namespace GeomDetEnumerators;

CSCChamberSpecs::CSCChamberSpecs( 
      const CSCGeometry* geom,
      int iChamberType, const TrapezoidalPlaneBounds& bounds,
      const CSCSpecsParcel& fupar,
      const CSCWireGroupPackage& wg )
  : GeomDetType( "CSC", CSC ), theChamberType( iChamberType ), theSpecsValues( fupar ),
    nstrips( static_cast<int>(specsValue(5)) ), stripDeltaPhi( specsValue(29) ), 
    centreToIntersectionOffset( specsValue(30) ), gangedStrips_( false )
  {
   LogTrace("CSCChamberSpecs|CSC") << myName << ": constructing specs for chamber " << 
    theName[iChamberType - 1] << ", type=" << iChamberType << ", this =" << this;
  
  // Reset gangedStrips_ for ME1A from config parameter
  if ( iChamberType == 1 ) gangedStrips_ = geom->gangedStrips(); 

  // Most wire info now comes from wire group section of DDD, but this doesn't...
  float wireAngleInDegrees = specsValue(12);
  
  // Related to strip geometry...
  float stripOffset1                       = specsValue(20);
  float stripOffset2                       = specsValue(21);
  float globalRadialPositionOfAlignmentPin = specsValue(24);
  float distanceFrameToAlignmentPin        = specsValue(25);
  float lengthOfChamber                    = specsValue(26);
  float distanceEndOfStripToAlignmentPin   = specsValue(27);
  float extentOfStripPlane                 = specsValue(28);

  // local y of alignment pin 
  float yAlignmentPin = -lengthOfChamber/2. + distanceFrameToAlignmentPin;

  // distance from alignment pin to symmetry centre of strip plane
  float alignmentPinToCentreOfStripPlane = distanceEndOfStripToAlignmentPin + extentOfStripPlane/2. ;

  // local y of symmetry centre of strip plane
  float yCentreOfStripPlane = yAlignmentPin  + alignmentPinToCentreOfStripPlane ;

  // distance from intersection of strips to symmetry centre of strip plane
  float whereStripsMeet = globalRadialPositionOfAlignmentPin + alignmentPinToCentreOfStripPlane ;

  // Possibly 'correct' distance to strip intersection
  //  if ( useCentreTIOffsets ) {
  //    float ctiOffset = this->ctiOffset();
  //    whereStripsMeet += ctiOffset; 
  //  }
  whereStripsMeet += centreToIntersectionOffset; // will have been reset to zero if not active

  // local y value of 1st wire in wire plane
  double yOfFirstWire = yAlignmentPin + wg.alignmentPinToFirstWire/10.; //@@ in cm

  // Build the unique LayerGeometry objects we require for each chamber type.
  // - There are 2 endcaps
  // - Alternate layers of strips are offset w.r.t. each other
  // - In ME11 the wire tilt angle needs to be a constant
  // global value; in the CMS local coordinates this is positive in +z
  // and negative in -z (magnitude 29 degrees as of 2002)

  // Thus we need 4 LGs differing in strip offset and wire angle

  float phiPitch = this->stripPhiPitch();

  // Layer thickness can come from specs too
  float hThickness = specsValue(32)/10./2.; // mm->cm, and then want half the thickness

  poszOddLayerGeometry = new CSCLayerGeometry( geom, iChamberType, bounds,
     nstrips, -stripOffset1, phiPitch, whereStripsMeet, extentOfStripPlane, yCentreOfStripPlane,
     wg, wireAngleInDegrees, yOfFirstWire, hThickness );

  poszEvenLayerGeometry = new CSCLayerGeometry( geom, iChamberType, bounds,
     nstrips, -stripOffset2, phiPitch, whereStripsMeet, extentOfStripPlane, yCentreOfStripPlane,
     wg, wireAngleInDegrees, yOfFirstWire, hThickness );

  negzOddLayerGeometry = new CSCLayerGeometry( geom, iChamberType, bounds,
     nstrips, -stripOffset1, phiPitch, whereStripsMeet, extentOfStripPlane, yCentreOfStripPlane,
     wg, -wireAngleInDegrees, yOfFirstWire, hThickness );

  negzEvenLayerGeometry = new CSCLayerGeometry( geom, iChamberType, bounds,
     nstrips, -stripOffset2, phiPitch, whereStripsMeet, extentOfStripPlane, yCentreOfStripPlane,
     wg, -wireAngleInDegrees, yOfFirstWire, hThickness );

}


CSCChamberSpecs::~CSCChamberSpecs()
{
  LogTrace("CSCChamberSpecs|CSC") << myName << " destroying this=" << this;

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
  // so use one gas gain value for all chambers (=300000)
  return 3.0e05;
}

float CSCChamberSpecs::chargePerCount() const {
  if(theChamberType <= 2) {
    return 0.25;
  } else {
    return 0.5;
  }
}

// STATICS

// Define the name of each chamber type
const std::string CSCChamberSpecs::theName[] =
  {"ME1/a", "ME1/b", "ME1/2", "ME1/3", "ME2/1", "ME2/2", "ME3/1", "ME3/2",
   "ME4/1", "ME4/2" };

// Define name of this class
const std::string CSCChamberSpecs::myName = "CSCChamberSpecs";
 
