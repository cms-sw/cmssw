/** \file
 *
 *  $Date: 2011/09/15 10:07:07 $
 *  $Revision: 1.8 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */
 

#include "Alignment/MuonAlignment/interface/AlignableDTBarrel.h"
#include "CondFormats/Alignment/interface/Alignments.h" 
#include "CondFormats/Alignment/interface/AlignmentErrors.h" 
#include "CondFormats/Alignment/interface/AlignmentSorter.h" 
#include "FWCore/MessageLogger/interface/MessageLogger.h"


/// The constructor simply copies the vector of wheels and computes the surface from them
AlignableDTBarrel::AlignableDTBarrel( const std::vector<AlignableDTWheel*> dtWheels ) 
   : AlignableComposite(dtWheels[0]->id(), align::AlignableDTBarrel)
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
void AlignableDTBarrel::dump( void ) const
{

  edm::LogInfo("AlignableDump") << (*this);
  for ( std::vector<AlignableDTWheel*>::const_iterator iWheel = theDTWheels.begin();
		iWheel != theDTWheels.end(); iWheel++ )
	(*iWheel)->dump();

}

//__________________________________________________________________________________________________
Alignments* AlignableDTBarrel::alignments( void ) const
{

  std::vector<Alignable*> comp = this->components();
  Alignments* m_alignments = new Alignments();
  // Add components recursively
  for ( std::vector<Alignable*>::iterator i=comp.begin(); i!=comp.end(); i++ )
    {
      Alignments* tmpAlignments = (*i)->alignments();
      std::copy( tmpAlignments->m_align.begin(), tmpAlignments->m_align.end(), 
				 std::back_inserter(m_alignments->m_align) );
	  delete tmpAlignments;
    }

  std::sort( m_alignments->m_align.begin(), m_alignments->m_align.end(), 
			 lessAlignmentDetId<AlignTransform>() );

  return m_alignments;

}

//__________________________________________________________________________________________________
AlignmentErrors* AlignableDTBarrel::alignmentErrors( void ) const
{

  std::vector<Alignable*> comp = this->components();
  AlignmentErrors* m_alignmentErrors = new AlignmentErrors();

  // Add components recursively
  for ( std::vector<Alignable*>::iterator i=comp.begin(); i!=comp.end(); i++ )
    {
	  AlignmentErrors* tmpAlignmentErrors = (*i)->alignmentErrors();
      std::copy( tmpAlignmentErrors->m_alignError.begin(), tmpAlignmentErrors->m_alignError.end(), 
				 std::back_inserter(m_alignmentErrors->m_alignError) );
	  delete tmpAlignmentErrors;
    }

  std::sort( m_alignmentErrors->m_alignError.begin(), m_alignmentErrors->m_alignError.end(), 
			 lessAlignmentDetId<AlignTransformError>() );

  return m_alignmentErrors;

}


