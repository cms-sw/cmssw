#include "Alignment/TrackerAlignment/interface/AlignablePxHalfBarrelLayer.h"

/// The constructor simply copies the vector of rods and computes the surface from its elements.
AlignablePxHalfBarrelLayer::AlignablePxHalfBarrelLayer( const std::vector<AlignableTrackerRod*> rods ) 
{

  theRods.insert( theRods.end(), rods.begin(), rods.end() );
  
  setSurface(computeSurface());

}


/// Clean delete of the vector and its elements
AlignablePxHalfBarrelLayer::~AlignablePxHalfBarrelLayer()
{

  for ( std::vector<AlignableTrackerRod*>::iterator iter = theRods.begin(); 
		iter != theRods.end(); iter++ ) 
    delete *iter;

}


/// Returns AlignableTrackerRod (rod) at given index
AlignableTrackerRod &AlignablePxHalfBarrelLayer::rod(int i)
{

  if (i >= size() ) 
	throw cms::Exception("LogicError") << "Rod index (" << i << ") out of range";

  return *theRods[i];

}


/// Returns surface corresponding to current position
/// and orientation, as given by average on all components
AlignableSurface AlignablePxHalfBarrelLayer::computeSurface()
{

  return AlignableSurface( computePosition(), computeOrientation() );

}


/// Compute average x position from all components (z and y forced to 0)
AlignablePxHalfBarrelLayer::PositionType AlignablePxHalfBarrelLayer::computePosition() 
{

  float xx = 0.;

  for ( std::vector<AlignableTrackerRod*>::iterator iRod=theRods.begin();
       iRod != theRods.end(); iRod++)
    xx += (*iRod)->globalPosition().x();

  xx /= static_cast<float>( theRods.size() );

  return PositionType( xx, 0.0, 0.0 );

}


/// Just initialize to default given by default constructor of a RotationType
AlignablePxHalfBarrelLayer::RotationType AlignablePxHalfBarrelLayer::computeOrientation() 
{

  return RotationType();

}



/// Twist layer by given angle (in radians)
void AlignablePxHalfBarrelLayer::twist(float rad) 
{

  float radOfRod;

  for ( std::vector<AlignableTrackerRod*>::iterator iter = theRods.begin();
		iter != theRods.end(); iter++ ) 
	{

	  //some local z-coordiates point inwards, others point outwards. 
	  //it points "outwards" if the scalar product of the global vector to 
	  //the rod origin with the local z-axis (given in global coordinates, 
	  //which is simply given by the last row of the orientation matrix) 
	  //is positiv... if it's not the same orientation you have to rotate
	  // in the other direction
	  GlobalVector lzaxis = ((*iter)->surface().toGlobal(LocalVector(0,0,1)));
	  // rotation axis
	  GlobalVector vec = ( (*iter)->globalPosition() - GlobalPoint(0,0,0) );
	  GlobalVector rotaxis( GlobalVector(vec.x(), vec.y(), 0).unit());
	  if ( ( rotaxis  * lzaxis) < 0. )  rotaxis *= -1.0 ;

	  radOfRod = rad * (*iter)->globalPosition().perp() / (*iter)->length();

	  (*iter)->rotateAroundGlobalAxis(rotaxis,radOfRod);

  }

}


/// Output layer information
std::ostream &operator << ( std::ostream &os, const AlignablePxHalfBarrelLayer & b )
{

  os << "  This PxHalfBarrelLayer contains " << b.theRods.size() << " Rods" << std::endl;
  os << "  (phi,r,z) = (" << b.globalPosition().phi() << "," 
     << b.globalPosition().perp() << "," << b.globalPosition().z();
  os << "), orientation" << std::endl<< b.globalRotation() << std::endl;
  return os;

}


/// Recursive printout of whole layer structure
void AlignablePxHalfBarrelLayer::dump( void )
{

  std::cout << (*this);
  for ( std::vector<AlignableTrackerRod*>::iterator iRod = theRods.begin();
		iRod != theRods.end(); iRod++ )
	std::cout << (**iRod);
	

}



