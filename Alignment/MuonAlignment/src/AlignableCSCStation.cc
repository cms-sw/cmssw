/** \file
 *
 *  $Date: 2011/09/15 10:07:07 $
 *  $Revision: 1.7 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */
 

#include "Alignment/MuonAlignment/interface/AlignableCSCStation.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



/// The constructor simply copies the vector of CSC Rings and computes the surface from them
AlignableCSCStation::AlignableCSCStation( const std::vector<AlignableCSCRing*> cscRings ) 
   : AlignableComposite(cscRings[0]->id(), align::AlignableCSCStation)
{

  theCSCRings.insert( theCSCRings.end(), cscRings.begin(), cscRings.end() );

  setSurface( computeSurface() );
   
}
      

/// Clean delete of the vector and its elements
AlignableCSCStation::~AlignableCSCStation() 
{
  for ( std::vector<AlignableCSCRing*>::iterator iter = theCSCRings.begin(); 
	iter != theCSCRings.end(); iter++)
    delete *iter;

}

/// Return Alignable CSC Ring at given index
AlignableCSCRing &AlignableCSCStation::ring(int i) 
{
  
  if (i >= size() ) 
	throw cms::Exception("LogicError") << "CSC Ring index (" << i << ") out of range";

  return *theCSCRings[i];
  
}


/// Returns surface corresponding to current position
/// and orientation, as given by average on all components
AlignableSurface AlignableCSCStation::computeSurface()
{

  return AlignableSurface( computePosition(), computeOrientation() );

}



/// Compute average z position from all components (x and y forced to 0)
AlignableCSCStation::PositionType AlignableCSCStation::computePosition()  
{

  float zz = 0.;

  for ( std::vector<AlignableCSCRing*>::iterator ilayer = theCSCRings.begin();
		ilayer != theCSCRings.end(); ilayer++ )
    zz += (*ilayer)->globalPosition().z();

  zz /= static_cast<float>(theCSCRings.size());

  return PositionType( 0.0, 0.0, zz );

}


/// Just initialize to default given by default constructor of a RotationType
AlignableCSCStation::RotationType AlignableCSCStation::computeOrientation() 
{
  return RotationType();
}


// /// Twists all components by given angle
// void AlignableCSCStation::twist(float rad) 
// {
//   for ( std::vector<AlignableCSCRing*>::iterator iter = theCSCRings.begin();
//            iter != theCSCRings.end(); iter++ )
//         (*iter)->twist(rad);

// }


/// Output Station information
std::ostream &operator << (std::ostream& os, const AlignableCSCStation& b )
{

  os << "This CSC Station contains " << b.theCSCRings.size() << " CSC rings" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," 
     << b.globalPosition().perp() << "," << b.globalPosition().z();
  os << "),  orientation:" << std::endl<< b.globalRotation() << std::endl;
  return os;

}


/// Recursive printout of whole CSC Station structure
void AlignableCSCStation::dump( void ) const
{

  edm::LogInfo("AlignableDump") << (*this);
  for ( std::vector<AlignableCSCRing*>::const_iterator iRing = theCSCRings.begin();
		iRing != theCSCRings.end(); iRing++ )
	 edm::LogInfo("AlignableDump")  << (**iRing);

}
