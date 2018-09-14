#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

// Original Author:  ?
//     Last Update:  Max Stark
//            Date:  Mon, 15 Feb 2016 09:32:12 CET

// alignment
#include "Alignment/TrackerAlignment/interface/AlignableTrackerBuilder.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"



//_____________________________________________________________________________
AlignableTracker
::AlignableTracker(const TrackerGeometry* trackerGeometry,
                   const TrackerTopology* trackerTopology) :
  // id not yet known, will be set by builder
  AlignableComposite(0, align::Tracker, RotationType()),
  tTopo_(trackerTopology),
  trackerNameSpace_(trackerTopology),
  alignableObjectId_(trackerGeometry, nullptr, nullptr)
{
  AlignableTrackerBuilder builder(trackerGeometry, trackerTopology);
  builder.buildAlignables(this);
  trackerNameSpace_ = builder.trackerNameSpace();
  alignableObjectId_ = builder.objectIdProvider();
}

//_____________________________________________________________________________
void AlignableTracker::update(const TrackerGeometry* trackerGeometry,
                              const TrackerTopology* trackerTopology)
{
  AlignableTrackerBuilder builder(trackerGeometry, trackerTopology);
  builder.buildAlignables(this, /* update = */ true);
}

//_____________________________________________________________________________
align::Alignables AlignableTracker::merge( const Alignables& list1,
                                           const Alignables& list2 ) const
{
  Alignables all = list1;

  all.insert( all.end(), list2.begin(), list2.end() );

  return all;
}

//_____________________________________________________________________________
Alignments* AlignableTracker::alignments( void ) const
{

  align::Alignables comp = this->components();
  Alignments* m_alignments = new Alignments();
  // Add components recursively
  for ( align::Alignables::iterator i=comp.begin(); i!=comp.end(); i++ )
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

//_____________________________________________________________________________
AlignmentErrorsExtended* AlignableTracker::alignmentErrors( void ) const
{

  align::Alignables comp = this->components();
  AlignmentErrorsExtended* m_alignmentErrors = new AlignmentErrorsExtended();

  // Add components recursively
  for ( align::Alignables::iterator i=comp.begin(); i!=comp.end(); i++ )
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
