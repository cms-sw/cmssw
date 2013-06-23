/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Andrew W. Rose                       ///
/// 2008                                 ///
/// ////////////////////////////////////////

#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerDetUnit.h"

StackedTrackerDetUnit::StackedTrackerDetUnit(){}

StackedTrackerDetUnit::StackedTrackerDetUnit( StackedTrackerDetId aStackId, 
					      const StackContents& listStackMembers )
  : StackId(aStackId),
    stackMembers(listStackMembers)
{}

StackedTrackerDetUnit::StackedTrackerDetUnit( const StackedTrackerDetUnit& aDetUnit )
  : StackId( aDetUnit.Id() ),
    stackMembers( aDetUnit.theStackMembers() )
{}

/// Method to return the Stack Member (argument = 0 means inner, argument = 1 means outer )
DetId StackedTrackerDetUnit::stackMember( unsigned int stackMemberIdentifier ) const
{
  if( stackMembers.find(stackMemberIdentifier) != stackMembers.end() )
  {
    return stackMembers.find(stackMemberIdentifier)->second;
  }
  return DetId(0);
}

