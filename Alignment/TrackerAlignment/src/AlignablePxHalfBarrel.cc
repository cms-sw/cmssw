#include "Alignment/TrackerAlignment/interface/AlignablePxHalfBarrel.h"


/// The constructor simply copies the vector of layers and computes the surface from them
AlignablePxHalfBarrel::AlignablePxHalfBarrel( const std::vector<AlignablePxHalfBarrelLayer*> 
											  barrelLayers )
{

  thePxHalfBarrelLayers.insert( thePxHalfBarrelLayers.end(), 
								barrelLayers.begin(), barrelLayers.end() );

  setSurface( computeSurface() );

}
  

/// Clean delete of the vector and its elements
AlignablePxHalfBarrel::~AlignablePxHalfBarrel() {

  for ( std::vector<AlignablePxHalfBarrelLayer*>::iterator iter = thePxHalfBarrelLayers.begin();
		iter != thePxHalfBarrelLayers.end(); iter++ ) 
    delete *iter;

}


/// Return AlignablePxBarrelLayer at given index
AlignablePxHalfBarrelLayer &AlignablePxHalfBarrel::layer( int i ) 
{

  if (i >= size() ) 
	throw cms::Exception("LogicError") << "Layer index (" << i << ") out of range";

  return *thePxHalfBarrelLayers[i];
  
}


/// Returns surface corresponding to current position
/// and orientation, as given by average on all components
AlignableSurface AlignablePxHalfBarrel::computeSurface()
{

  return AlignableSurface( computePosition(), computeOrientation() );

}


/// Compute average x position from all components (z and y forced to 0)
AlignablePxHalfBarrel::PositionType AlignablePxHalfBarrel::computePosition() 
{

  float xx = 0.;

  for ( std::vector<AlignablePxHalfBarrelLayer*>::iterator iLayer = thePxHalfBarrelLayers.begin();
        iLayer != thePxHalfBarrelLayers.end(); iLayer++ )
    xx += (*iLayer)->globalPosition().x();

  xx /= static_cast<float>( thePxHalfBarrelLayers.size() );

  return PositionType( xx, 0.0, 0.0 );

}


/// Just initialize to default given by default constructor of a RotationType
AlignablePxHalfBarrel::RotationType AlignablePxHalfBarrel::computeOrientation() 
{

  return RotationType();

}


/// Twists all components by given angle
void AlignablePxHalfBarrel::twist(float rad) 
{
  
  for ( std::vector<AlignablePxHalfBarrelLayer*>::iterator iter = thePxHalfBarrelLayers.begin();
		iter != thePxHalfBarrelLayers.end(); iter++ ) 
	(*iter)->twist(rad); 
  
}


/// Output Half Barrel information
std::ostream& operator << (std::ostream& os, const AlignablePxHalfBarrel& b )
{

  os << "This PxHalfBarrel contains " << b.thePxHalfBarrelLayers.size() 
	 << " Barrel layers" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," 
     << b.globalPosition().perp() << "," << b.globalPosition().z();
  os << "),  orientation:" << std::endl<< b.globalRotation() << std::endl;
  return os;

}


/// Recursive printout of whole Half Barrel structure
void AlignablePxHalfBarrel::dump( void )
{

  std::cout << (*this);
  for ( std::vector<AlignablePxHalfBarrelLayer*>::iterator iLayer = thePxHalfBarrelLayers.begin();
		iLayer != thePxHalfBarrelLayers.end(); iLayer++ )
	(*iLayer)->dump();
  
}



