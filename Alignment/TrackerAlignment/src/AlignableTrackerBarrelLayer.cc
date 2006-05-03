#include "Alignment/TrackerAlignment/interface/AlignableTrackerRod.h"
#include "Alignment/TrackerAlignment/interface/AlignableTrackerBarrelLayer.h"


//--------------------------------------------------------------------------------------------------
AlignableTrackerBarrelLayer::AlignableTrackerBarrelLayer( const std::vector<AlignableTrackerRod*> rods ) 
{

  theRods.insert( theRods.end(), rods.begin(), rods.end() );

  setSurface( computeSurface() );

}


//--------------------------------------------------------------------------------------------------
AlignableTrackerBarrelLayer::~AlignableTrackerBarrelLayer()
{

  for ( std::vector<AlignableTrackerRod*>::iterator iter = theRods.begin(); 
		iter != theRods.end(); iter++) 
	delete *iter;
  
}



//--------------------------------------------------------------------------------------------------
std::vector<Alignable*> AlignableTrackerBarrelLayer::components() const
{
  std::vector<Alignable*> result; 
  result.insert( result.end(), theRods.begin(), theRods.end());
  return result;
}


//--------------------------------------------------------------------------------------------------
AlignableTrackerRod& AlignableTrackerBarrelLayer::rod(int i)
{

  if (i >= size() ) 
	throw cms::Exception("LogicError") << "Rod index (" << i << ") out of range";

  return *theRods[i];

}


//--------------------------------------------------------------------------------------------------
AlignableSurface AlignableTrackerBarrelLayer::computeSurface()
{

  return AlignableSurface( computePosition(), computeOrientation() );

}


//--------------------------------------------------------------------------------------------------
AlignableTrackerBarrelLayer::PositionType AlignableTrackerBarrelLayer::computePosition() 
{

  float zz = 0.;

  for (std::vector<AlignableTrackerRod*>::iterator irod=theRods.begin();
       irod != theRods.end(); irod++)
	  zz += (*irod)->globalPosition().z();

  zz /= static_cast<float>( theRods.size() );

  return PositionType( 0.0, 0.0, zz );

}


//--------------------------------------------------------------------------------------------------
AlignableTrackerBarrelLayer::RotationType AlignableTrackerBarrelLayer::computeOrientation() 
{

  return RotationType();

}


//--------------------------------------------------------------------------------------------------
/// A BarrelLayer ist twisted by rotating each Rod around the original center
/// (before any "mis-alignment"... e.g. the nominal position...
/// here you have to watch out! once the nominal position might include 
/// already some "aligned" detector)
/// and with the orientation of +/- its original local z-axis. Furthermore
/// the rotation angle is calculated from the rod length... which currently 
/// is simply calculated from the GeomDetUnits on the rod... and NOT from 
/// the distance between the two supporting barrel disks... which would be 
/// more correct...
void AlignableTrackerBarrelLayer::twist(float rad) 
{

  float radOfRod;

  for ( std::vector<AlignableTrackerRod*>::iterator iter = theRods.begin();
		iter != theRods.end(); iter++ ) 
	{

	  // Some local z-coordiates point inwards, others point outwards. 
	  // It points "outwards" if the scalar product of the global vector to 
	  // the rod origin with the local z-axis (given in global coordinates, 
	  // which is simply given by the last row of the orientation matrix) 
	  // is positive... if it's not the same orientation you have to rotate
	  // in the other direction
	  GlobalVector lzaxis = ((*iter)->surface().toGlobal(LocalVector(0,0,1)));
	  // rotation axis
	  GlobalVector vec = ( (*iter)->globalPosition() - GlobalPoint(0,0,0) );
	  GlobalVector rotaxis( GlobalVector(vec.x(), vec.y(), 0).unit());

	  if (  (rotaxis * lzaxis) < 0. ) rotaxis *= -1.0 ;

	  radOfRod = rad * (*iter)->globalPosition().perp() / (*iter)->length();

	  (*iter)->rotateAroundGlobalAxis(rotaxis,radOfRod);
	  
	}

}


//--------------------------------------------------------------------------------------------------
std::ostream &operator << (std::ostream &os, const AlignableTrackerBarrelLayer & b ){

  os << "  This BarrelLayer contains " << b.theRods.size() << " Rods" << std::endl;
  os << "  (phi,r,z) = (" << b.globalPosition().phi() << "," 
     << b.globalPosition().perp() << "," << b.globalPosition().z() << std::endl;
  os << "),  orientation:" << std::endl<< b.globalRotation() << std::endl;
  return os;

}


//--------------------------------------------------------------------------------------------------
void AlignableTrackerBarrelLayer::dump( void )
{

  std::cout << (*this);
  for ( std::vector<AlignableTrackerRod*>::iterator iRod = theRods.begin();
		iRod != theRods.end(); iRod++ )
	std::cout << (**iRod);
	

}










