#include "Alignment/TrackerAlignment/interface/AlignablePixelHalfBarrelLayer.h"

//--------------------------------------------------------------------------------------------------
AlignablePixelHalfBarrelLayer::AlignablePixelHalfBarrelLayer( const std::vector<AlignableTrackerRod*> rods ) 
{

  theRods.insert( theRods.end(), rods.begin(), rods.end() );
  
  setSurface(computeSurface());

}


//--------------------------------------------------------------------------------------------------
AlignablePixelHalfBarrelLayer::~AlignablePixelHalfBarrelLayer()
{

  for ( std::vector<AlignableTrackerRod*>::iterator iter = theRods.begin(); 
		iter != theRods.end(); iter++ ) 
    delete *iter;

}


//--------------------------------------------------------------------------------------------------
std::vector<Alignable*> AlignablePixelHalfBarrelLayer::components() const 
{
  std::vector<Alignable*> result; 
  result.insert( result.end(), theRods.begin(), theRods.end());
  return result;
}


//--------------------------------------------------------------------------------------------------
AlignableTrackerRod &AlignablePixelHalfBarrelLayer::rod(int i)
{

  if (i >= size() ) 
	throw cms::Exception("LogicError") << "Rod index (" << i << ") out of range";

  return *theRods[i];

}


//--------------------------------------------------------------------------------------------------
AlignableSurface AlignablePixelHalfBarrelLayer::computeSurface()
{

  return AlignableSurface( computePosition(), computeOrientation() );

}


//--------------------------------------------------------------------------------------------------
AlignablePixelHalfBarrelLayer::PositionType AlignablePixelHalfBarrelLayer::computePosition() 
{

  float xx = 0.;

  for ( std::vector<AlignableTrackerRod*>::iterator iRod=theRods.begin();
       iRod != theRods.end(); iRod++)
    xx += (*iRod)->globalPosition().x();

  xx /= static_cast<float>( theRods.size() );

  return PositionType( xx, 0.0, 0.0 );

}


//--------------------------------------------------------------------------------------------------
AlignablePixelHalfBarrelLayer::RotationType AlignablePixelHalfBarrelLayer::computeOrientation() 
{

  return RotationType();

}



//--------------------------------------------------------------------------------------------------
/// A PixelHalfBarrelLayer ist twisted by rotating each AlignableTrackerRod
/// around the original center (before any "mis-alignment".. e.g. the nominal 
/// position...here you have to watch out!  once the nomnal position might include 
/// already some "aligned" detector) and with the orientation of +/- its original 
/// local z-axis. Furthermore the rotation angle is calculated from the rod 
/// length....which currently is simply calculated from the detunits on 
/// the rod... and NOT from the distance between the two supporting barrel
/// disks....which would be more correct...
void AlignablePixelHalfBarrelLayer::twist( float radians ) 
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

	  radOfRod = radians * (*iter)->globalPosition().perp() / (*iter)->length();

	  (*iter)->rotateAroundGlobalAxis(rotaxis,radOfRod);

  }

}


//--------------------------------------------------------------------------------------------------
std::ostream &operator << ( std::ostream &os, const AlignablePixelHalfBarrelLayer & b )
{

  os << "  This PixelHalfBarrelLayer contains " << b.theRods.size() << " Rods" << std::endl;
  os << "  (phi,r,z) = (" << b.globalPosition().phi() << "," 
     << b.globalPosition().perp() << "," << b.globalPosition().z();
  os << "), orientation" << std::endl<< b.globalRotation() << std::endl;
  return os;

}


//--------------------------------------------------------------------------------------------------
void AlignablePixelHalfBarrelLayer::dump( void )
{

  std::cout << (*this);
  for ( std::vector<AlignableTrackerRod*>::iterator iRod = theRods.begin();
		iRod != theRods.end(); iRod++ )
	std::cout << (**iRod);
	

}



