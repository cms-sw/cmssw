/** \file
 *
 *  $Date: 2011/09/15 10:07:07 $
 *  $Revision: 1.3 $
 *  \author Jim Pivarski - Texas A&M University
 */
 

#include "Alignment/MuonAlignment/interface/AlignableCSCRing.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



/// The constructor simply copies the vector of CSC Chambers and computes the surface from them
AlignableCSCRing::AlignableCSCRing( const std::vector<AlignableCSCChamber*> cscChambers ) 
   : AlignableComposite(cscChambers[0]->id(), align::AlignableCSCRing)
{

  theCSCChambers.insert( theCSCChambers.end(), cscChambers.begin(), cscChambers.end() );

  setSurface( computeSurface() );
   
}
      

/// Clean delete of the vector and its elements
AlignableCSCRing::~AlignableCSCRing() 
{
  for ( std::vector<AlignableCSCChamber*>::iterator iter = theCSCChambers.begin(); 
	iter != theCSCChambers.end(); iter++)
    delete *iter;

}

/// Return Alignable CSC Chamber at given index
AlignableCSCChamber &AlignableCSCRing::chamber(int i) 
{
  
  if (i >= size() ) 
	throw cms::Exception("LogicError") << "CSC Chamber index (" << i << ") out of range";

  return *theCSCChambers[i];
  
}


/// Returns surface corresponding to current position
/// and orientation, as given by average on all components
AlignableSurface AlignableCSCRing::computeSurface()
{

  return AlignableSurface( computePosition(), computeOrientation() );

}



/// Compute average z position from all components (x and y forced to 0)
AlignableCSCRing::PositionType AlignableCSCRing::computePosition()  
{

  float zz = 0.;

  for ( std::vector<AlignableCSCChamber*>::iterator ichamber = theCSCChambers.begin();
		ichamber != theCSCChambers.end(); ichamber++ )
    zz += (*ichamber)->globalPosition().z();

  zz /= static_cast<float>(theCSCChambers.size());

  return PositionType( 0.0, 0.0, zz );

}


/// Just initialize to default given by default constructor of a RotationType
AlignableCSCRing::RotationType AlignableCSCRing::computeOrientation() 
{
  return RotationType();
}


// /// Twists all components by given angle
// void AlignableCSCRing::twist(float rad) 
// {
//   for ( std::vector<AlignableCSCChamber*>::iterator iter = theCSCChambers.begin();
//            iter != theCSCChambers.end(); iter++ )
//         (*iter)->twist(rad);

// }


/// Output Ring information
std::ostream &operator << (std::ostream& os, const AlignableCSCRing& b )
{

  os << "This CSC Ring contains " << b.theCSCChambers.size() << " CSC chambers" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," 
     << b.globalPosition().perp() << "," << b.globalPosition().z();
  os << "),  orientation:" << std::endl<< b.globalRotation() << std::endl;
  return os;

}


/// Recursive printout of whole CSC Ring structure
void AlignableCSCRing::dump( void ) const
{

  edm::LogInfo("AlignableDump") << (*this);
  for ( std::vector<AlignableCSCChamber*>::const_iterator iChamber = theCSCChambers.begin();
		iChamber != theCSCChambers.end(); iChamber++ )
	 edm::LogInfo("AlignableDump")  << (**iChamber);

}
