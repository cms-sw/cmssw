#include "Alignment/TrackerAlignment/interface/AlignableTIDLayer.h"
#include "Alignment/TrackerAlignment/interface/AlignableTIDRing.h"

//--------------------------------------------------------------------------------------------------
AlignableTIDLayer::AlignableTIDLayer( const std::vector<AlignableTIDRing*> rings )
{
  
  theRings.insert( theRings.end(), rings.begin(), rings.end() );

  setSurface(computeSurface());

}


//--------------------------------------------------------------------------------------------------
AlignableTIDLayer::~AlignableTIDLayer()
{
  for ( std::vector<AlignableTIDRing*>::iterator iter = theRings.begin(); 
		iter != theRings.end(); iter++) 
    delete *iter;

}


//--------------------------------------------------------------------------------------------------
std::vector<Alignable*> AlignableTIDLayer::components() const 
{
  std::vector<Alignable*> result; 
  result.insert( result.end(), theRings.begin(), theRings.end() );
  return result;
}


//--------------------------------------------------------------------------------------------------
AlignableTIDRing &AlignableTIDLayer::ring(int i)
{

  if (i >= size() )
	throw cms::Exception("LogicError") << "Ring index (" << i << ") out of range";

  return *theRings[i];

}


//--------------------------------------------------------------------------------------------------
AlignableSurface AlignableTIDLayer::computeSurface()
{

  return AlignableSurface( computePosition(), computeOrientation() );

}


//--------------------------------------------------------------------------------------------------
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


//--------------------------------------------------------------------------------------------------
AlignableTIDLayer::RotationType AlignableTIDLayer::computeOrientation() 
{

  return RotationType();

}


//--------------------------------------------------------------------------------------------------
std::ostream &operator << (std::ostream &os, const AlignableTIDLayer & b )
{

  os << "  This TIDLayer contains " << b.theRings.size() << " Rings" << std::endl;
  os << "  (phi, r, z) =  (" << b.globalPosition().phi() << "," 
     << b.globalPosition().perp() << " , " << b.globalPosition().z();
  os << "),  orientation:" << std::endl<< b.globalRotation() << std::endl;
  return os;

}


//--------------------------------------------------------------------------------------------------
void AlignableTIDLayer::dump( void )
{

  std::cout << (*this);
  for ( std::vector<AlignableTIDRing*>::iterator iRing = theRings.begin();
		iRing != theRings.end(); iRing++ )
	std::cout << (**iRing);
	

}







