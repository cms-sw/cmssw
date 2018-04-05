#include "Geometry/CSCGeometry/interface/CSCWireTopology.h"

#include "Geometry/CSCGeometry/src/CSCGangedWireGrouping.h"
#include "Geometry/CSCGeometry/src/CSCNonslantedWireGeometry.h"
#include "Geometry/CSCGeometry/src/CSCSlantedWireGeometry.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <cmath>

CSCWireTopology::~CSCWireTopology() { 
  delete theWireGrouping;
  delete theWireGeometry; 
}

CSCWireTopology::CSCWireTopology( 
		     const CSCWireGroupPackage& wg,
		     double yOfFirstWire,
                     float wireAngleInDegrees ) :
      theWireGrouping( nullptr ), theWireGeometry( nullptr ),
      theAlignmentPinToFirstWire( wg.alignmentPinToFirstWire / 10. ) {

     // Pass consecutiveGroups and wiresInConsecutiveGroups
     // directly on to theWireGrouping ctor. These vectors
     // are transient info and can go once ctor is complete.
         
  theWireGrouping = new CSCGangedWireGrouping( wg.consecutiveGroups,
		           wg.wiresInEachGroup, wg.numberOfGroups );

  const float zeroprecision = 1.E-06; // blur zero a bit, for comparisons

  float wireAngleInRadians = wireAngleInDegrees*degree;

  //@@ Conversion from mm to cm
  float wireSpacing = wg.wireSpacing   / 10.; // in cm
  float nw = wg.narrowWidthOfWirePlane / 10.; // in cm
  float ww = wg.wideWidthOfWirePlane   / 10.; // in cm
  float lw = wg.lengthOfWirePlane      / 10.; // in cm

   LogTrace("CSCWireTopology|CSC") <<  
       "CSCWireTopology constructing CSCWireGeometry with:\n" <<
       " wireSpacing = " << wireSpacing*10. << " (mm) " <<
       ", yOfFirstWire = " << yOfFirstWire << " (cm) " << 
     ", wireAngle = " << wireAngleInDegrees << " (deg) = " << wireAngleInRadians << " (rads)" <<
     ", extent: n, w, l = " << nw << ", " << ww << ", " << lw << " (cm)";

  if ( fabs(wireAngleInDegrees) > zeroprecision ) { 
    theWireGeometry = new CSCSlantedWireGeometry( wireSpacing, yOfFirstWire, nw, ww, lw, wireAngleInRadians );
  }
  else {
    theWireGeometry = new CSCNonslantedWireGeometry( wireSpacing, yOfFirstWire, nw, ww, lw );
  }
}

CSCWireTopology::CSCWireTopology( const CSCWireTopology& mewt ) :
  theAlignmentPinToFirstWire(mewt.theAlignmentPinToFirstWire) {
  if (mewt.theWireGrouping) theWireGrouping = mewt.theWireGrouping->clone();
  else theWireGrouping = nullptr;
  if (mewt.theWireGeometry) theWireGeometry = mewt.theWireGeometry->clone();
  else theWireGeometry = nullptr;
  
}

CSCWireTopology& CSCWireTopology::operator=( const CSCWireTopology& mewt ) {
  if ( &mewt != this ) {
    delete theWireGrouping;
    if ( mewt.theWireGrouping ) 
      theWireGrouping = mewt.theWireGrouping->clone();
    else 
      theWireGrouping = nullptr;
    
    delete theWireGeometry;
    if ( mewt.theWireGeometry ) 
      theWireGeometry = mewt.theWireGeometry->clone();
    else 
      theWireGeometry = nullptr;

    theAlignmentPinToFirstWire = mewt.theAlignmentPinToFirstWire;
    
  }
  return *this;
}

LocalPoint CSCWireTopology::localPosition( const MeasurementPoint& ) const {
  edm::LogWarning("CSC") << "CSCWireTopology: localPosition unimplemented. Don't use it." << "\n";
  return LocalPoint();
}

LocalError CSCWireTopology::localError( const MeasurementPoint&, 
					  const MeasurementError& ) const {
  edm::LogWarning("CSC") << "CSCWireTopology: localError unimplemented. Don't use it." << "\n";
  return LocalError();
}

MeasurementPoint CSCWireTopology::measurementPosition( 
				       const LocalPoint& ) const {
  edm::LogWarning("CSC") << "CSCWireTopology: measurementPosition unimplemented. Don't use it." << "\n";
  return MeasurementPoint();
}

MeasurementError CSCWireTopology::measurementError( const LocalPoint&, 
					   const LocalError& ) const {
  edm::LogWarning("CSC") << "CSCWireTopology: measurementError unimplemented. Don't use it." << "\n";
  return MeasurementError();
}

int CSCWireTopology::channel( const LocalPoint& p ) const {
  int wire = theWireGeometry->nearestWire( p );
  int group = theWireGrouping->wireGroup( wire );
  return group; // 0 means out of range or dead region
}

float CSCWireTopology::yOfWireGroup(int wireGroup, float x) const {
   float wire = middleWireOfGroup( wireGroup );
   return theWireGeometry->yOfWire( wire, x );
}
 
float CSCWireTopology::yResolution( int wireGroup ) const {
  // Return the resolution in the wire group measurement (in cm).
  // To really be resolution in local y this should be scaled by Sin
  // factors involving strip and wire angles, and also add a contribution
  // from strip measurement (which is negligible w.r.t. wire resolution.)
  // These factors can increase the number here by up to 1.26.

  // Should be improvable for smaller partial wire groups

  return wireSpacing() * theWireGrouping->numberOfWiresPerGroup( wireGroup ) / sqrt(12.);
}

std::pair<float, float> CSCWireTopology::equationOfWire( float wire ) const {
  return theWireGeometry->equationOfWire( wire );
}

float CSCWireTopology::restrictToYOfWirePlane( float y ) const {
  // Reset y to lie within bounds of wire plane at top and bottom.
  
  std::pair<float, float> ylim = theWireGeometry->yLimitsOfWirePlane();

  if ( y < ylim.first ) {
    y = ylim.first;
  }
  else if ( y > ylim.second ) {
    y = ylim.second;
  }
  return y;
}

bool CSCWireTopology::insideYOfWirePlane( float y ) const {
  // Returns true if arg falls within y limits of wire plane; false otherwise.
      
  std::pair<float, float> ylim = theWireGeometry->yLimitsOfWirePlane();

  if ( y < ylim.first ) {
    return false;
  }
  else if ( y > ylim.second ) {
    return false;
  }
  return true;
}
