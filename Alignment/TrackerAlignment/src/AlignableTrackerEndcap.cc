#include "Alignment/TrackerAlignment/interface/AlignableTrackerEndcap.h"


/// The constructor simply copies the vector of layers and computes the surface from them
AlignableTrackerEndcap::AlignableTrackerEndcap( const std::vector<AlignableTrackerEndcapLayer*> endcapLayers )  
{

  theEndcapLayers.insert( theEndcapLayers.end(), endcapLayers.begin(), endcapLayers.end() );

  setSurface( computeSurface() );

}


/// Clean delete of the vector and its elements
AlignableTrackerEndcap::~AlignableTrackerEndcap() 
{

  for ( std::vector<AlignableTrackerEndcapLayer*>::iterator iter = theEndcapLayers.begin(); 
		iter != theEndcapLayers.end(); iter++) 
    delete *iter;

}


/// Return AlignableTrackerEndcapLayer at given index
AlignableTrackerEndcapLayer &AlignableTrackerEndcap::layer(int i) 
{

  if (i >= size() ) 
	throw cms::Exception("LogicError") << "Layer index (" << i << ") out of range";

  return *theEndcapLayers[i];
  
}


/// Returns surface corresponding to current position
/// and orientation, as given by average on all components
AlignableSurface AlignableTrackerEndcap::computeSurface()
{

  return AlignableSurface( computePosition(), computeOrientation() );

}


/// Compute average z position from all components (x and y forced to 0)
AlignableTrackerEndcap::PositionType AlignableTrackerEndcap::computePosition() 
{

  float zz=0.;

  for ( std::vector<AlignableTrackerEndcapLayer*>::iterator ilayer = theEndcapLayers.begin();
	   ilayer != theEndcapLayers.end(); ilayer++ )
	zz += (*ilayer)->globalPosition().z();

  zz /= static_cast<float>( theEndcapLayers.size() );

  return PositionType( 0.0, 0.0, zz );

}


/// Just initialize to default given by default constructor of a RotationType
AlignableTrackerEndcap::RotationType AlignableTrackerEndcap::computeOrientation() 
{

  return RotationType();

}


/// Output Half Barrel information
std::ostream &operator << (std::ostream& os, const AlignableTrackerEndcap& b )
{

  os << "This Endcap contains " << b.theEndcapLayers.size() << " Endcap layers" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," 
     << b.globalPosition().perp() << "," << b.globalPosition().z();
  os << "),  orientation:" << std::endl<< b.globalRotation() << std::endl;
  return os;

}


/// Recursive printout of whole Half Endcap structure
void AlignableTrackerEndcap::dump( void )
{

  // Print the whole structure
  
  std::cout << (*this);
  for ( std::vector<AlignableTrackerEndcapLayer*>::iterator iLayer = theEndcapLayers.begin();
		iLayer != theEndcapLayers.end(); iLayer++ )
	(*iLayer)->dump();

}





