/** \file
 *
 *  $Date: 2008/04/10 16:36:41 $
 *  $Revision: 1.7 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */


#include <memory>

#include "Alignment/MuonAlignment/interface/AlignableCSCEndcap.h"
#include "CondFormats/Alignment/interface/Alignments.h" 
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h" 
#include "FWCore/MessageLogger/interface/MessageLogger.h"


/// The constructor simply copies the vector of stations and computes the surface from them
AlignableCSCEndcap::AlignableCSCEndcap( const std::vector<AlignableCSCStation*>& cscStations ) 
   : AlignableComposite(cscStations[0]->id(), align::AlignableCSCEndcap)
{

  theCSCStations.insert( theCSCStations.end(), cscStations.begin(), cscStations.end() );

  // maintain also list of components
  for (const auto& station: cscStations) {
    const auto mother = station->mother();
    this->addComponent(station); // components will be deleted by dtor of AlignableComposite
    station->setMother(mother); // restore previous behaviour where mother is not set
  }

  setSurface( computeSurface() );
  compConstraintType_ = Alignable::CompConstraintType::POSITION_Z;
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
  Alignments* m_alignments = new Alignments();

  // Add components recursively
  for (const auto& i: this->components()) {
    std::unique_ptr<Alignments> tmpAlignments{i->alignments()};
    std::copy(tmpAlignments->m_align.begin(), tmpAlignments->m_align.end(),
              std::back_inserter(m_alignments->m_align));
  }

  // sort by rawId
  std::sort( m_alignments->m_align.begin(), m_alignments->m_align.end());

  return m_alignments;
}

//__________________________________________________________________________________________________

AlignmentErrorsExtended* AlignableCSCEndcap::alignmentErrors( void ) const
{
  AlignmentErrorsExtended* m_alignmentErrors = new AlignmentErrorsExtended();

  // Add components recursively
  for (const auto& i: this->components()) {
    std::unique_ptr<AlignmentErrorsExtended> tmpAlignmentErrorsExtended{i->alignmentErrors()};
    std::copy(tmpAlignmentErrorsExtended->m_alignError.begin(), tmpAlignmentErrorsExtended->m_alignError.end(),
              std::back_inserter(m_alignmentErrors->m_alignError) );
  }

  // sort by rawId
  std::sort( m_alignmentErrors->m_alignError.begin(), m_alignmentErrors->m_alignError.end());

  return m_alignmentErrors;
}

