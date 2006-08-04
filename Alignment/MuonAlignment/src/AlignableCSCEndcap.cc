/** \file
 *
 *  $Date: 2006/8/4 10:10:07 $
 *  $Revision: 1.0 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */
 

#include "Alignment/MuonAlignment/interface/AlignableCSCEndcap.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


/// The constructor simply copies the vector of stations and computes the surface from them
AlignableCSCEndcap::AlignableCSCEndcap( const std::vector<AlignableCSCStation*> cscStations ) 
{

  theCSCStations.insert( theCSCStations.end(), cscStations.begin(), cscStations.end() );

  setSurface( computeSurface() );
   
}
      

/// Clean delete of the vector and its elements
AlignableCSCEndcap::~AlignableCSCEndcap() 
{
  for ( std::vector<AlignableCSCStation*>::iterator iter = theCSCStations.begin(); 
	iter != theCSCStations.end(); iter++)
    delete *iter;

}

/// Return AlignableCSCEndcap station at given index
AlignableCSCStation &AlignableCSCEndcap::station(int i) 
{
  
  if (i >= size() ) 
	throw cms::Exception("LogicError") << "Station index (" << i << ") out of range";

  return *theCSCStations[i];
  
}


/// Returns surface corresponding to current position
/// and orientation, as given by average on all components
AlignableSurface AlignableCSCEndcap::computeSurface()
{

  return AlignableSurface( computePosition(), computeOrientation() );

}



/// Compute average z position from all components (x and y forced to 0)
AlignableCSCEndcap::PositionType AlignableCSCEndcap::computePosition() 
{

  float zz = 0.;

  for ( std::vector<AlignableCSCStation*>::iterator ilayer = theCSCStations.begin();
		ilayer != theCSCStations.end(); ilayer++ )
    zz += (*ilayer)->globalPosition().z();

  zz /= static_cast<float>(theCSCStations.size());

  return PositionType( 0.0, 0.0, zz );

}


/// Just initialize to default given by default constructor of a RotationType
AlignableCSCEndcap::RotationType AlignableCSCEndcap::computeOrientation() 
{
  return RotationType();
}


/// Output Half Barrel information
std::ostream &operator << (std::ostream& os, const AlignableCSCEndcap& b )
{

  os << "This EndCap contains " << b.theCSCStations.size() << " CSC stations" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," 
     << b.globalPosition().perp() << "," << b.globalPosition().z();
  os << "),  orientation:" << std::endl<< b.globalRotation() << std::endl;
  return os;

}


/// Recursive printout of whole Half Barrel structure
void AlignableCSCEndcap::dump( void )
{

  edm::LogInfo("AlignableDump") << (*this);
  for ( std::vector<AlignableCSCStation*>::iterator iLayer = theCSCStations.begin();
		iLayer != theCSCStations.end(); iLayer++ )
	(*iLayer)->dump();

}
