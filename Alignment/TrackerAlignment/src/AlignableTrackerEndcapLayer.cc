#include "Alignment/TrackerAlignment/interface/AlignableTrackerPetal.h"
#include "Alignment/TrackerAlignment/interface/AlignableTrackerEndcapLayer.h"

//__________________________________________________________________________________________________
AlignableTrackerEndcapLayer::AlignableTrackerEndcapLayer(  const std::vector<AlignableTrackerPetal*> petals ) 
{

  thePetals.insert( thePetals.end(), petals.begin(), petals.end() );

  setSurface( computeSurface() );

}


//__________________________________________________________________________________________________
AlignableTrackerEndcapLayer::~AlignableTrackerEndcapLayer()
{

  for ( std::vector<AlignableTrackerPetal*>::iterator iter = thePetals.begin(); 
	iter != thePetals.end(); iter++)
    delete *iter;

}


//__________________________________________________________________________________________________
std::vector<Alignable*> AlignableTrackerEndcapLayer::components() const 
{
  std::vector<Alignable*> result; 
  result.insert( result.end(), thePetals.begin(), thePetals.end());
  return result;
}


//__________________________________________________________________________________________________
AlignableTrackerPetal &AlignableTrackerEndcapLayer::petal(int i)
{

  if (i >= size() ) 
	throw cms::Exception("LogicError") << "Petal index (" << i << ") out of range";

  return *thePetals[i];

}


//__________________________________________________________________________________________________
AlignableSurface AlignableTrackerEndcapLayer::computeSurface()
{

  return AlignableSurface( computePosition(), computeOrientation() );

}



//__________________________________________________________________________________________________
AlignableTrackerEndcapLayer::PositionType AlignableTrackerEndcapLayer::computePosition() 
{

  float zz = 0.;

  for (std::vector<AlignableTrackerPetal*>::iterator ipetal=thePetals.begin();
	   ipetal != thePetals.end(); ipetal++)
      zz += (*ipetal)->globalPosition().z();

  zz /= static_cast<float>( thePetals.size() );

  return PositionType( 0.0, 0.0, zz );

}


//__________________________________________________________________________________________________
AlignableTrackerEndcapLayer::RotationType AlignableTrackerEndcapLayer::computeOrientation() 
{

  return RotationType();

}


//__________________________________________________________________________________________________
std::ostream &operator << ( std::ostream &os, const AlignableTrackerEndcapLayer & b )
{

  os << "  This EndcapLayer contains " << b.thePetals.size() << " Petals" << std::endl;
  os << "  (phi, r, z) = " << b.globalPosition().phi() << "," 
     << b.globalPosition().perp() << "," << b.globalPosition().z() << std::endl;
  os << "), orientation:" << std::endl<< b.globalRotation() << std::endl;
  return os;

}


//__________________________________________________________________________________________________
void AlignableTrackerEndcapLayer::dump( void )
{

  std::cout << (*this);
  for ( std::vector<AlignableTrackerPetal*>::iterator iPetal = thePetals.begin();
		iPetal != thePetals.end(); iPetal++ )
	std::cout << (**iPetal);
	
}







