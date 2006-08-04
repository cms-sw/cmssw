/** \file
 *
 *  $Date: 2006/8/4 10:10:07 $
 *  $Revision: 1.0 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */
 

#include "Alignment/MuonAlignment/interface/AlignableDTBarrel.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


/// The constructor simply copies the vector of wheels and computes the surface from them
AlignableDTBarrel::AlignableDTBarrel( const std::vector<AlignableDTWheel*> dtWheels ) 
{

  theDTWheels.insert( theDTWheels.end(), dtWheels.begin(), dtWheels.end() );

  setSurface( computeSurface() );
   
}
      

/// Clean delete of the vector and its elements
AlignableDTBarrel::~AlignableDTBarrel() 
{
  for ( std::vector<AlignableDTWheel*>::iterator iter = theDTWheels.begin(); 
	iter != theDTWheels.end(); iter++)
    delete *iter;

}

/// Return AlignableBarrelLayer at given index
AlignableDTWheel &AlignableDTBarrel::wheel(int i) 
{
  
  if (i >= size() ) 
	throw cms::Exception("LogicError") << "Wheel index (" << i << ") out of range";

  return *theDTWheels[i];
  
}


/// Returns surface corresponding to current position
/// and orientation, as given by average on all components
AlignableSurface AlignableDTBarrel::computeSurface()
{

  return AlignableSurface( computePosition(), computeOrientation() );

}



/// Compute average z position from all components (x and y forced to 0)
AlignableDTBarrel::PositionType AlignableDTBarrel::computePosition() 
{

  float zz = 0.;

  for ( std::vector<AlignableDTWheel*>::iterator ilayer = theDTWheels.begin();
		ilayer != theDTWheels.end(); ilayer++ )
    zz += (*ilayer)->globalPosition().z();

  zz /= static_cast<float>(theDTWheels.size());

  return PositionType( 0.0, 0.0, zz );

}


/// Just initialize to default given by default constructor of a RotationType
AlignableDTBarrel::RotationType AlignableDTBarrel::computeOrientation() 
{
  return RotationType();
}



/// Output Half Barrel information
std::ostream &operator << (std::ostream& os, const AlignableDTBarrel& b )
{

  os << "This DTBarrel contains " << b.theDTWheels.size() << " Barrel wheels" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," 
     << b.globalPosition().perp() << "," << b.globalPosition().z();
  os << "),  orientation:" << std::endl<< b.globalRotation() << std::endl;
  return os;

}


/// Recursive printout of whole Half Barrel structure
void AlignableDTBarrel::dump( void )
{

  edm::LogInfo("AlignableDump") << (*this);
  for ( std::vector<AlignableDTWheel*>::iterator iWheel = theDTWheels.begin();
		iWheel != theDTWheels.end(); iWheel++ )
	(*iWheel)->dump();

}
