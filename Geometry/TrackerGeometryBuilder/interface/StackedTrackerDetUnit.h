/*! \class   StackedTrackerDetUnit
 *  \brief   GeomDetUnit-derived class for Pt modules
 *  \details
 *
 *  \author Andrew W. Rose
 *  \author Ivan Reid
 *  \date   2008
 *
 */

#ifndef STACKED_TRACKER_DET_UNIT_H
#define STACKED_TRACKER_DET_UNIT_H

#include "DataFormats/SiPixelDetId/interface/StackedTrackerDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <map>
#include <vector>

class StackedTrackerDetUnit {
public:
  typedef std::map < unsigned int, DetId >                 StackContents;
  typedef std::map < unsigned int, DetId >::const_iterator StackContentsIterator;

  /// Constructors
  StackedTrackerDetUnit();
  StackedTrackerDetUnit( StackedTrackerDetId aStackId, const StackContents& listStackMembers );
  StackedTrackerDetUnit( StackedTrackerDetId aStackId, const StackContents& listStackMembers,
                         int detUnitWindow, std::vector< std::vector< int > > offsetArray );
  StackedTrackerDetUnit( const StackedTrackerDetUnit& aDetUnit );

  /// Methods for data members
  int size()                                   { return stackMembers.size(); }
  DetId stackMember( unsigned int stackMemberIdentifier ) const;
  StackedTrackerDetId Id() const               { return StackId; }
  const StackContents& theStackMembers() const { return stackMembers; }

  /// CBC3 dedicated methods
  /// everything is in half-strip units
  const int detUnitWindow() const                                   { return CBC3_WindowSize; }
  const int asicOffset( int asicNumber, int partitionNumber ) const { return (modulePartitionOffsets.at(asicNumber)).at(partitionNumber); }

private:

  /// Data members
  StackedTrackerDetId StackId;
  StackContents       stackMembers;  

  /// CBC3 data members
  /// everything is in half-strip units
  int                               CBC3_WindowSize;
  std::vector< std::vector< int > > modulePartitionOffsets;

  /// Array of offsets, needed for copy constructor
  const std::vector< std::vector< int > > offsetArray() const { return modulePartitionOffsets; }

};

#endif

