#include "Alignment/TrackerAlignment/interface/AlignableTIDLayer.h"
#include "Alignment/TrackerAlignment/interface/AlignableTIDRing.h"

/// The constructor simply copies the vector of rings and computes the surface from them
AlignableTIDLayer::AlignableTIDLayer( const std::vector<AlignableTIDRing*> rings )
{
  
  theRings.insert( theRings.end(), rings.begin(), rings.end() );

  setSurface(computeSurface());

}


/// Clean delete of the vector and its elements
AlignableTIDLayer::~AlignableTIDLayer()
{
  for ( std::vector<AlignableTIDRing*>::iterator iter = theRings.begin(); 
		iter != theRings.end(); iter++) 
    delete *iter;

}


/// Return AlignableRod at given index
AlignableTIDRing &AlignableTIDLayer::ring(int i)
{

  if (i >= size() )
	throw cms::Exception("LogicError") << "Ring index (" << i << ") out of range";

  return *theRings[i];

}


/// Returns surface corresponding to current position
/// and orientation, as given by average on all components
AlignableSurface AlignableTIDLayer::computeSurface()
{

  return AlignableSurface( computePosition(), computeOrientation() );

}


/// Compute average z position from all components (x and y forced to 0)
AlignableTIDLayer::PositionType AlignableTIDLayer::computePosition() 
{

  // put the layer in on the beam Axis and at the average z of the Rings
  float zz=0.;

  for (std::vector<AlignableTIDRing*>::iterator iring=theRings.begin();
       iring != theRings.end(); iring++)
    zz += (*iring)->globalPosition().z();

  zz /= static_cast<float>(theRings.size());
  
  return PositionType( 0.0, 0.0, zz );

}


/// Just initialize to default given by default constructor of a RotationType
AlignableTIDLayer::RotationType AlignableTIDLayer::computeOrientation() 
{

  return RotationType();

}


/// Output layer information
std::ostream &operator << (std::ostream &os, const AlignableTIDLayer & b )
{

  os << "  This TIDLayer contains " << b.theRings.size() << " Rings" << std::endl;
  os << "  (phi, r, z) =  (" << b.globalPosition().phi() << "," 
     << b.globalPosition().perp() << " , " << b.globalPosition().z();
  os << "),  orientation:" << std::endl<< b.globalRotation() << std::endl;
  return os;

}


/// Recursive printout of whole layer structure
void AlignableTIDLayer::dump( void )
{

  std::cout << (*this);
  for ( std::vector<AlignableTIDRing*>::iterator iRing = theRings.begin();
		iRing != theRings.end(); iRing++ )
	std::cout << (**iRing);
	

}







