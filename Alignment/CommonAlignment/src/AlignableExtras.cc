/** \file AlignableExtras
 *
 *  Original author: Andreas Mussgiller, August 2010
 *
 *  $Date: 2010/09/10 10:26:20 $
 *  $Revision: 1.1 $
 *  (last update by $Author: mussgill $)
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

// Alignment
#include "Alignment/CommonAlignment/interface/AlignableBeamSpot.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"

#include "Alignment/CommonAlignment/interface/AlignableExtras.h"

//__________________________________________________________________________________________________
AlignableExtras::AlignableExtras()
{
  align::Alignables& alis = alignableLists_.get("BeamSpot");
  alis.push_back(new AlignableBeamSpot());
  components_.push_back(alis.back());
}

//__________________________________________________________________________________________________
void AlignableExtras::dump( void ) const
{
  Alignables comp = this->components();

  // Dump this
  edm::LogInfo("AlignableDump") 
    << " AlignableExtras knows " << comp.size() << " alignable(s)" << std::endl;

  // Dump components
  for ( Alignables::iterator i=comp.begin(); i!=comp.end(); ++i )
    (*i)->dump();
}

//__________________________________________________________________________________________________
Alignments* AlignableExtras::alignments( void ) const
{
  align::Alignables comp = this->components();
  Alignments* m_alignments = new Alignments();
  // Add components recursively
  for ( align::Alignables::iterator i=comp.begin(); i!=comp.end(); ++i )
    {
      Alignments* tmpAlignments = (*i)->alignments();
      std::copy( tmpAlignments->m_align.begin(), tmpAlignments->m_align.end(), 
		 std::back_inserter(m_alignments->m_align) );
	  delete tmpAlignments;
    }

  // sort by rawId
  std::sort( m_alignments->m_align.begin(), m_alignments->m_align.end());

  return m_alignments;
}

//__________________________________________________________________________________________________
AlignmentErrorsExtended* AlignableExtras::alignmentErrors( void ) const
{
  align::Alignables comp = this->components();
  AlignmentErrorsExtended* m_alignmentErrors = new AlignmentErrorsExtended();

  // Add components recursively
  for ( align::Alignables::iterator i=comp.begin(); i!=comp.end(); ++i )
    {
	  AlignmentErrorsExtended* tmpAlignmentErrorsExtended = (*i)->alignmentErrors();
      std::copy( tmpAlignmentErrorsExtended->m_alignError.begin(), tmpAlignmentErrorsExtended->m_alignError.end(), 
		 std::back_inserter(m_alignmentErrors->m_alignError) );
	  delete tmpAlignmentErrorsExtended;
    }
  
  // sort by rawId
  std::sort( m_alignmentErrors->m_alignError.begin(), m_alignmentErrors->m_alignError.end());

  return m_alignmentErrors;
}

//______________________________________________________________________________
void AlignableExtras::initializeBeamSpot(double x, double y, double z,
					 double dxdz, double dydz)
{
  align::Alignables& alis = beamSpot();
  AlignableBeamSpot * aliBS = dynamic_cast<AlignableBeamSpot*>(alis.back());
  if (aliBS) {
    aliBS->initialize(x, y, z, dxdz, dydz);
  } else {
    edm::LogError("AlignableExtras") 
      << " AlignableBeamSpot not available. Cannot initialize!" << std::endl;
  }
}

//______________________________________________________________________________
void AlignableExtras::resetBeamSpot()
{
  align::Alignables& alis = beamSpot();
  AlignableBeamSpot * aliBS = dynamic_cast<AlignableBeamSpot*>(alis.back());
  if (aliBS) {
    aliBS->reset();
  } else {
    edm::LogWarning("AlignableExtras")
      << "@SUB=AlignableExtras::resetBeamSpot"
      << "AlignableBeamSpot not available. Cannot reset!" << std::endl;
  }
}
