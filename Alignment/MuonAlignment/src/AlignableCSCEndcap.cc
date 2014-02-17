/** \file
 *
 *  $Date: 2011/09/15 09:40:22 $
 *  $Revision: 1.8 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */
 
 
#include "Alignment/MuonAlignment/interface/AlignableCSCEndcap.h"
#include "CondFormats/Alignment/interface/Alignments.h" 
#include "CondFormats/Alignment/interface/AlignmentErrors.h" 
#include "CondFormats/Alignment/interface/AlignmentSorter.h" 
#include "FWCore/MessageLogger/interface/MessageLogger.h"


/// The constructor simply copies the vector of stations and computes the surface from them
AlignableCSCEndcap::AlignableCSCEndcap( const std::vector<AlignableCSCStation*> cscStations ) 
   : AlignableComposite(cscStations[0]->id(), align::AlignableCSCEndcap)
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
void AlignableCSCEndcap::dump( void ) const
{

  edm::LogInfo("AlignableDump") << (*this);
  for ( std::vector<AlignableCSCStation*>::const_iterator iLayer = theCSCStations.begin();
		iLayer != theCSCStations.end(); iLayer++ )
	(*iLayer)->dump();

}

//__________________________________________________________________________________________________

Alignments* AlignableCSCEndcap::alignments( void ) const
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

AlignmentErrors* AlignableCSCEndcap::alignmentErrors( void ) const
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

