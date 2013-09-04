
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/

#ifndef STACKED_TRACKER_DET_UNIT_H
#define STACKED_TRACKER_DET_UNIT_H

#include "DataFormats/SiPixelDetId/interface/StackedTrackerDetId.h"


#include "DataFormats/DetId/interface/DetId.h"
#include <map>

class StackedTrackerDetUnit {
public:
  typedef	std::map < unsigned int , DetId >	StackContents;
  typedef	std::map < unsigned int , DetId >::const_iterator StackContentsIterator;


  StackedTrackerDetUnit();
  StackedTrackerDetUnit( StackedTrackerDetId aStackId, const StackContents& listStackMembers );
  StackedTrackerDetUnit( const StackedTrackerDetUnit& aDetUnit );

  int size(){return stackMembers.size();}
  DetId stackMember(unsigned int stackMemberIdentifier) const;

  StackedTrackerDetId Id() const {return StackId;}
  const StackContents& theStackMembers() const {return stackMembers;}

private:
  StackedTrackerDetId StackId;
  StackContents stackMembers;  
};

#endif // Tracker_StackedTrackerDetUnit_H

