/** \file
 *
 *  $Date: 2006/8/4 10:10:07 $
 *  $Revision: 1.0 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */
 

#include "Alignment/MuonAlignment/interface/AlignableCSCStation.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



/// The constructor simply copies the vector of CSC Chambers and computes the surface from them
AlignableCSCStation::AlignableCSCStation( const std::vector<AlignableCSCChamber*> dtChambers ) 
{

  theCSCChambers.insert( theCSCChambers.end(), dtChambers.begin(), dtChambers.end() );

  setSurface( computeSurface() );
   
}
      

/// Clean delete of the vector and its elements
AlignableCSCStation::~AlignableCSCStation() 
{
  for ( std::vector<AlignableCSCChamber*>::iterator iter = theCSCChambers.begin(); 
	iter != theCSCChambers.end(); iter++)
    delete *iter;

}

/// Return Alignable CSC Chamber at given index
AlignableCSCChamber &AlignableCSCStation::chamber(int i) 
{
  
  if (i >= size() ) 
	throw cms::Exception("LogicError") << "CSC Chamber index (" << i << ") out of range";

  return *theCSCChambers[i];
  
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

  for ( std::vector<AlignableCSCChamber*>::iterator ilayer = theCSCChambers.begin();
		ilayer != theCSCChambers.end(); ilayer++ )
    zz += (*ilayer)->globalPosition().z();

  zz /= static_cast<float>(theCSCChambers.size());

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
//   for ( std::vector<AlignableCSCChamber*>::iterator iter = theCSCChambers.begin();
//            iter != theCSCChambers.end(); iter++ )
//         (*iter)->twist(rad);

// }


/// Output Station information
std::ostream &operator << (std::ostream& os, const AlignableCSCStation& b )
{

  os << "This CSC Station contains " << b.theCSCChambers.size() << " CSC chambers" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," 
     << b.globalPosition().perp() << "," << b.globalPosition().z();
  os << "),  orientation:" << std::endl<< b.globalRotation() << std::endl;
  return os;

}


/// Recursive printout of whole CSC Station structure
void AlignableCSCStation::dump( void )
{

  edm::LogInfo("AlignableDump") << (*this);
  for ( std::vector<AlignableCSCChamber*>::iterator iChamber = theCSCChambers.begin();
		iChamber != theCSCChambers.end(); iChamber++ )
	 edm::LogInfo("AlignableDump")  << (**iChamber);

}
