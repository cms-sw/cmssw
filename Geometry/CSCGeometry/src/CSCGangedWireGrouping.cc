#include "Geometry/CSCGeometry/src/CSCGangedWireGrouping.h"

#include <iostream>
#include <iomanip>
#include <algorithm>

CSCGangedWireGrouping::CSCGangedWireGrouping( 
  const Container& consecutiveGroups, 
  const Container& wiresInConsecutiveGroups, 
  int numberOfGroups ) : 
     theNumberOfGroups(numberOfGroups) {

  int countGroups = 0;       // counter for no. of wire groups
  int firstWire = 1;         // (virtual) wire number which is 1st in a group
  int countConsecutive = 0;  // counter for the sections in DDD

  for ( CIterator it = consecutiveGroups.begin();
	it != consecutiveGroups.end(); ++it ) {
    int igs = *it;
    if ( igs != 0 ) {
      // igs is number of consecutive groups each containing
      // an identical number of wires
      countGroups += igs;
      for ( int ic = 0; ic != igs; ++ic ) {
        theFirstWireOfEachWireGroup.push_back( firstWire );
        int wiresInGroup= wiresInConsecutiveGroups[countConsecutive];
        theNumberOfWiresPerWireGroup.push_back( wiresInGroup );
        firstWire += wiresInGroup; // ready for next group
      }
    }
    else {
      // zero means a gap - just add in the no. of virtual wires in gap
      firstWire += wiresInConsecutiveGroups[countConsecutive];
    }
    ++countConsecutive; // ready for next set of consecutive groups
  }

  theNumberOfWires = firstWire - 1; // at end of loop we're on 1st for next

  if ( countGroups != numberOfGroups ) {
    std::cout << "CSCGangedWireGrouping: ERROR in parsing wire info from DDD..." << std::endl;
    std::cout << "groups expected = " << numberOfGroups << 
      " groups seen = " << countGroups << std::endl;
    std::cout << "Please report this error to Tim.Cox@cern.ch" << std::endl;
  }
  //  if ( CSCGeometryLogger::debugV ) {
  if ( true ) { //@@ FIXME of course
    std::cout << "CSCGangedWireGrouping constructor complete," <<
       " wire group list follows... " << std::endl;
    std::cout << "Size of group buffers = " << theFirstWireOfEachWireGroup.size() <<
      " and " << theNumberOfWiresPerWireGroup.size() << std::endl;
    std::cout << " wg#    1st wire  #wires" << std::endl;
    for ( size_t i = 0; i != theFirstWireOfEachWireGroup.size(); ++i ) {
       std::cout << std::setw(4) << i+1 << std::setw(12) << theFirstWireOfEachWireGroup[i] <<
	 std::setw(8) << theNumberOfWiresPerWireGroup[i] << std::endl;
    }
    std::cout << "Total no. of wires = " << theNumberOfWires << std::endl;
  }
}

int CSCGangedWireGrouping::numberOfWiresPerGroup( int wireGroup ) const { 
  if ( wireGroup > 0 && wireGroup <= theNumberOfGroups )
    return theNumberOfWiresPerWireGroup[ wireGroup-1 ];
  else return 0;
}

int CSCGangedWireGrouping::wireGroup(int wire) const {
  // Return wire group number (start at 1) containing (virtual) 'wire'
  // Return 0 if 'wire' is not in a wire group; this includes
  // wires out outside the chamber, and lying in 'dead' regions

  int wireG = 0;

  // upper_bound works on a sorted range and points to first element 
  // succeeding supplied value.

  CIterator it = upper_bound( theFirstWireOfEachWireGroup.begin(),
			 theFirstWireOfEachWireGroup.end(), wire );

  // The '-1' steps back to group in which the wire lies

  int id = it - theFirstWireOfEachWireGroup.begin() - 1;

  // Skip id=-1 (i.e. value is less than any in range)

  if ( id >= 0 ) { // index of group in which wire lies
    int wiresInGroup = theNumberOfWiresPerWireGroup[id];
    int firstWire = theFirstWireOfEachWireGroup[id];
  
  // Require wire not past end of group (may be in a dead region, or
  // bigger than total wires in chamber)
    if ( wire < (firstWire + wiresInGroup) ) wireG = id + 1;
  }
  return wireG;
}

float CSCGangedWireGrouping::middleWireOfGroup( int wireGroup ) const {
  // Return exact wire number for group with odd number of wires
  // Return half-integer wire number for group with even number of wires
  // i.e. first + 0.5 * float(#) - 0.5
  // Return 0 if out of range

  float middleWire = 0.;
  if ( wireGroup > 0 && wireGroup <= theNumberOfGroups ) {
    middleWire = theFirstWireOfEachWireGroup[ wireGroup-1 ] +
      0.5 * theNumberOfWiresPerWireGroup[ wireGroup-1 ] -
         0.5 ;
  }
  return middleWire;
}

