#include "RecoTauTag/TauTagTools/interface/ECALBounds.h"

#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
 
//Ported from ORCA

static const float epsilon = 0.001; // should not matter at all
 
static BoundCylinder* initCylinder()
{
  Surface::RotationType rot; // unit rotation matrix
  auto cb = new SimpleCylinderBounds(ECALBounds::barrel_innerradius() - epsilon, 
	       		      ECALBounds::barrel_innerradius() + epsilon, 
	       		     -ECALBounds::barrel_halfLength(), 
	       		     +ECALBounds::barrel_halfLength());
  return new BoundCylinder( Cylinder::computeRadius(*cb), Surface::PositionType(0,0,0), rot, cb );
}

static BoundDisk* initNegativeDisk()
{
  Surface::RotationType rot; // unit rotation matrix
  
  return new BoundDisk( Surface::PositionType( 0, 0, -ECALBounds::endcap_innerZ() ), rot, 
			new SimpleDiskBounds( 0, ECALBounds::endcap_outerradius(), -epsilon, epsilon ) );
}
 
static BoundDisk* initPositiveDisk()
{
  Surface::RotationType rot; // unit rotation matrix
  
  return new BoundDisk( Surface::PositionType( 0, 0, +ECALBounds::endcap_innerZ() ), rot, 
			new SimpleDiskBounds( 0, ECALBounds::endcap_outerradius(), -epsilon, epsilon ) );
}

// static initializers 

const ReferenceCountingPointer<BoundCylinder> ECALBounds::theCylinder     = initCylinder();
const ReferenceCountingPointer<BoundDisk>     ECALBounds::theNegativeDisk = initNegativeDisk();
const ReferenceCountingPointer<BoundDisk>     ECALBounds::thePositiveDisk = initPositiveDisk();
