#include "Alignment/TrackerAlignment/interface/AlignableTrackerPetal.h"
#include "Alignment/TrackerAlignment/interface/AlignableTrackerEndcapLayer.h"

/// The constructor simply copies the vector of petals and computes the surface from its elements.
AlignableTrackerEndcapLayer::AlignableTrackerEndcapLayer(  const std::vector<AlignableTrackerPetal*> petals ) 
{

  thePetals.insert( thePetals.end(), petals.begin(), petals.end() );

  setSurface( computeSurface() );

}


/// Clean delete of the vector and its elements
AlignableTrackerEndcapLayer::~AlignableTrackerEndcapLayer()
{

  for ( std::vector<AlignableTrackerPetal*>::iterator iter = thePetals.begin(); 
	iter != thePetals.end(); iter++)
    delete *iter;

}


/// Returns AlignableTrackerPetal at given index
AlignableTrackerPetal &AlignableTrackerEndcapLayer::petal(int i)
{

  if (i >= size() ) 
	throw cms::Exception("LogicError") << "Petal index (" << i << ") out of range";

  return *thePetals[i];

}


/// Returns surface corresponding to current position
AlignableSurface AlignableTrackerEndcapLayer::computeSurface()
{

  return AlignableSurface( computePosition(), computeOrientation() );

}



/// Compute average z position from all components (x and y forced to 0)
AlignableTrackerEndcapLayer::PositionType AlignableTrackerEndcapLayer::computePosition() 
{

  float zz = 0.;

  for (std::vector<AlignableTrackerPetal*>::iterator ipetal=thePetals.begin();
	   ipetal != thePetals.end(); ipetal++)
      zz += (*ipetal)->globalPosition().z();

  zz /= static_cast<float>( thePetals.size() );

  return PositionType( 0.0, 0.0, zz );

}


/// Just initialize to default given by default constructor of a RotationType
AlignableTrackerEndcapLayer::RotationType AlignableTrackerEndcapLayer::computeOrientation() 
{

  return RotationType();

}


/// Output layer information
std::ostream &operator << ( std::ostream &os, const AlignableTrackerEndcapLayer & b )
{

  os << "  This EndcapLayer contains " << b.thePetals.size() << " Petals" << std::endl;
  os << "  (phi, r, z) = " << b.globalPosition().phi() << "," 
     << b.globalPosition().perp() << "," << b.globalPosition().z() << std::endl;
  os << "), orientation:" << std::endl<< b.globalRotation() << std::endl;
  return os;

}


/// Recursive printout of whole layer structure
void AlignableTrackerEndcapLayer::dump( void )
{

  std::cout << (*this);
  for ( std::vector<AlignableTrackerPetal*>::iterator iPetal = thePetals.begin();
		iPetal != thePetals.end(); iPetal++ )
	std::cout << (**iPetal);
	
}







