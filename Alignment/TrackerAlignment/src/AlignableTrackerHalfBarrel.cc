#include "Alignment/TrackerAlignment/interface/AlignableTrackerHalfBarrel.h"


/// The constructor simply copies the vector of layers and computes the surface from them
AlignableTrackerHalfBarrel::AlignableTrackerHalfBarrel
( const std::vector<AlignableTrackerBarrelLayer*> barrelLayers ) 
{

  theBarrelLayers.insert( theBarrelLayers.end(), barrelLayers.begin(), barrelLayers.end() );

  setSurface( computeSurface() );
   
}
      

/// Clean delete of the vector and its elements
AlignableTrackerHalfBarrel::~AlignableTrackerHalfBarrel() 
{
  for ( std::vector<AlignableTrackerBarrelLayer*>::iterator iter = theBarrelLayers.begin(); 
	iter != theBarrelLayers.end(); iter++)
    delete *iter;

}

/// Return AlignableTrackerBarrelLayer at given index
AlignableTrackerBarrelLayer &AlignableTrackerHalfBarrel::layer(int i) 
{
  
  if (i >= size() ) 
	throw cms::Exception("LogicError") << "Layer index (" << i << ") out of range";

  return *theBarrelLayers[i];
  
}


/// Returns surface corresponding to current position
/// and orientation, as given by average on all components
AlignableSurface AlignableTrackerHalfBarrel::computeSurface()
{

  return AlignableSurface( computePosition(), computeOrientation() );

}



/// Compute average z position from all components (x and y forced to 0)
AlignableTrackerHalfBarrel::PositionType AlignableTrackerHalfBarrel::computePosition() 
{

  float zz = 0.;

  for ( std::vector<AlignableTrackerBarrelLayer*>::iterator ilayer = theBarrelLayers.begin();
		ilayer != theBarrelLayers.end(); ilayer++ )
    zz += (*ilayer)->globalPosition().z();

  zz /= static_cast<float>(theBarrelLayers.size());

  return PositionType( 0.0, 0.0, zz );

}


/// Just initialize to default given by default constructor of a RotationType
AlignableTrackerHalfBarrel::RotationType AlignableTrackerHalfBarrel::computeOrientation() 
{
  return RotationType();
}


/// Twists all components by given angle
void AlignableTrackerHalfBarrel::twist(float rad) 
{

  for ( std::vector<AlignableTrackerBarrelLayer*>::iterator iter = theBarrelLayers.begin();
	   iter != theBarrelLayers.end(); iter++ ) 
	(*iter)->twist(rad);
  
}




/// Output Half Barrel information
std::ostream &operator << (std::ostream& os, const AlignableTrackerHalfBarrel& b )
{

  os << "This HalfBarrel contains " << b.theBarrelLayers.size() << " Barrel layers" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," 
     << b.globalPosition().perp() << "," << b.globalPosition().z();
  os << "),  orientation:" << std::endl<< b.globalRotation() << std::endl;
  return os;

}


/// Recursive printout of whole Half Barrel structure
void AlignableTrackerHalfBarrel::dump( void )
{

  std::cout << (*this);
  for ( std::vector<AlignableTrackerBarrelLayer*>::iterator iLayer = theBarrelLayers.begin();
		iLayer != theBarrelLayers.end(); iLayer++ )
	(*iLayer)->dump();

}
