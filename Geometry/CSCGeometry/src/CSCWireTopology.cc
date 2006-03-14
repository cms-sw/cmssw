#include "Geometry/CSCGeometry/interface/CSCWireTopology.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"

#include "Geometry/CSCGeometry/src/CSCGangedWireGrouping.h"
#include "Geometry/CSCGeometry/src/CSCNonslantedWireGeometry.h"
#include "Geometry/CSCGeometry/src/CSCSlantedWireGeometry.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "CLHEP/Units/SystemOfUnits.h"

#include <cmath>

CSCWireTopology::~CSCWireTopology() { 
  delete theWireGrouping;
  delete theWireGeometry; 
}

CSCWireTopology::CSCWireTopology( 
		     const CSCWireGroupPackage& wg,
		     double yOfFirstWire,
                     float wireAngleInDegrees ) :
      theWireGrouping( 0 ), theWireGeometry( 0 ),
      theAlignmentPinToFirstWire( wg.alignmentPinToFirstWire ) {

     // Pass consecutiveGroups and wiresInConsecutiveGroups
     // directly on to theWireGrouping ctor. These vectors
     // are transient info and can go once ctor is complete.
         
  theWireGrouping = new CSCGangedWireGrouping( wg.consecutiveGroups,
		           wg.wiresInEachGroup, wg.numberOfGroups );

  float wireAngleInRadians = wireAngleInDegrees*degree;
  float wireSpacing = wg.wireSpacing/10.; // in cm
  const float zeroprecision = 1.E-06; // blur zero a bit, for comparisons

  //@@ Set a value for yOfFirstWire if we haven't been given a real value.
  //@@ Recognized by a sentinel value > 0 since every 'real' value
  //@@ should be negative. I'm using the chamber type so I can identify
  //@@ ME1A and ME11 for special treatment (they share a common overall
  //@@ wire geometry.)

  // Values equivalent to ORCA_8_8_0 and earlier will also be negative
  // and for them we just use the value passed in from MELG, just as
  // for real values directly from the DDD.

  // odd no. of wires:  position central wire on local x axis
  // even no. of wires: position pair of central wires centred on local x axis

  if ( yOfFirstWire > 0. ) {
    // This is distance from one edge to symmetry centre of the set of
    // parallel wires
    float offsetOrthogonalToWires = 
         -0.5 * wireSpacing * (theWireGrouping->numberOfWires() - 1);
    // For ME1/1 I have even less idea what to do... symmetrize distance along y
    if ( fabs(wireAngleInDegrees) > zeroprecision ) {
      // I know distance from strip split to first wire along y axis
      // is 484.354 mm (from Petr Moissenz 07-Apr-2005 email)
      const float splitToFirstWire = 48.4354; // CMS uses cm 
      //@@ Warning: following values from DDD but wire geometry drawing
      //@@ implies both are exactly 1.0 cm longer (active gas volumes 1 cm shorter?)
      const float halfHeightInner = 21.5;
      const float halfHeightOuter = 52.75;
      // Require yOfFirstWire w.r.t. local origin
      if ( yOfFirstWire < 1.5 ) {
        yOfFirstWire = -splitToFirstWire + halfHeightInner; // ME1A
      }
      else {
        yOfFirstWire = -splitToFirstWire - halfHeightOuter; // ME11
      }
    }
    else yOfFirstWire = offsetOrthogonalToWires; // all other chambers
  }

  LogDebug("CSC") <<  
       "Constructing WireGeometry with:\n" <<
       " wireSpacing = " << wireSpacing*10. << " (mm) " <<
      ", yOfFirstWire = " << yOfFirstWire << " (cm) " <<
       ", wireAngle = " << wireAngleInDegrees << " (degrees) " <<
    wireAngleInRadians << " (radians)" << "\n";

  if ( fabs(wireAngleInDegrees) > zeroprecision ) { 
    theWireGeometry = new CSCSlantedWireGeometry( wireSpacing, yOfFirstWire, 
			  wireAngleInRadians );
  }
  else {
    theWireGeometry = new CSCNonslantedWireGeometry( wireSpacing, yOfFirstWire );
  }
}

CSCWireTopology::CSCWireTopology( const CSCWireTopology& mewt ) :
  theAlignmentPinToFirstWire(mewt.theAlignmentPinToFirstWire) {
  if (mewt.theWireGrouping) theWireGrouping = mewt.theWireGrouping->clone();
  if (mewt.theWireGeometry) theWireGeometry = mewt.theWireGeometry->clone();
  
}

CSCWireTopology& CSCWireTopology::operator=( const CSCWireTopology& mewt ) {
  if ( &mewt != this ) {
    delete theWireGrouping;
    if ( mewt.theWireGrouping ) 
      theWireGrouping = mewt.theWireGrouping->clone();
    else 
      theWireGrouping = 0;
    
    delete theWireGeometry;
    if ( mewt.theWireGeometry ) 
      theWireGeometry = mewt.theWireGeometry->clone();
    else 
      theWireGeometry = 0;

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

