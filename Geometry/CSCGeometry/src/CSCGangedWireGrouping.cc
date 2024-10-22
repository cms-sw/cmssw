#include <Geometry/CSCGeometry/src/CSCGangedWireGrouping.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstddef>

CSCGangedWireGrouping::CSCGangedWireGrouping(const Container& consecutiveGroups,
                                             const Container& wiresInConsecutiveGroups,
                                             int numberOfGroups)
    : theNumberOfGroups(numberOfGroups) {
  int countGroups = 0;       // counter for no. of wire groups
  int firstWire = 1;         // (virtual) wire number which is 1st in a group
  int countConsecutive = 0;  // counter for the sections in DDD

  for (int igs : consecutiveGroups) {
    if (igs != 0) {
      // igs is number of consecutive groups each containing
      // an identical number of wires
      countGroups += igs;
      for (int ic = 0; ic != igs; ++ic) {
        theFirstWireOfEachWireGroup.emplace_back(firstWire);
        int wiresInGroup = wiresInConsecutiveGroups[countConsecutive];
        theNumberOfWiresPerWireGroup.emplace_back(wiresInGroup);
        firstWire += wiresInGroup;  // ready for next group
      }
    } else {
      // zero means a gap - just add in the no. of virtual wires in gap
      firstWire += wiresInConsecutiveGroups[countConsecutive];
    }
    ++countConsecutive;  // ready for next set of consecutive groups
  }

  theNumberOfWires = firstWire - 1;  // at end of loop we're on 1st for next

  if (countGroups != numberOfGroups) {
    edm::LogError("CSC") << "CSCGangedWireGrouping: ERROR in parsing wire info from DDD..."
                         << "\n";
    edm::LogError("CSC") << "groups expected = " << numberOfGroups << " groups seen = " << countGroups << "\n";
    edm::LogError("CSC") << "Please report this error to Tim.Cox@cern.ch"
                         << "\n";
  }

  LogTrace("CSCWireGeometry|CSC") << "CSCGangedWireGrouping constructor complete,"
                                  << " wire group list follows... ";
  LogTrace("CSCWireGeometry|CSC") << "Size of group buffers = " << theFirstWireOfEachWireGroup.size() << " and "
                                  << theNumberOfWiresPerWireGroup.size();
  LogTrace("CSCWireGeometry|CSC") << " wg#    1st wire  #wires";
  for (size_t i = 0; i != theFirstWireOfEachWireGroup.size(); ++i) {
    LogTrace("CSCWireGeometry|CSC") << std::setw(4) << i + 1 << std::setw(12) << theFirstWireOfEachWireGroup[i]
                                    << std::setw(8) << theNumberOfWiresPerWireGroup[i];
  }
  LogTrace("CSCWireGeometry|CSC") << "Total no. of wires = " << theNumberOfWires;
}

int CSCGangedWireGrouping::numberOfWiresPerGroup(int wireGroup) const {
  if (wireGroup > 0 && wireGroup <= theNumberOfGroups)
    return theNumberOfWiresPerWireGroup[wireGroup - 1];
  else
    return 0;
}

int CSCGangedWireGrouping::wireGroup(int wire) const {
  // Return wire group number (start at 1) containing (virtual) 'wire'
  // Return 0 if 'wire' is not in a wire group; this includes
  // wires out outside the chamber, and lying in 'dead' regions

  int wireG = 0;

  // upper_bound works on a sorted range and points to first element
  // _succeeding_ supplied value.

  CIterator it = upper_bound(theFirstWireOfEachWireGroup.begin(), theFirstWireOfEachWireGroup.end(), wire);

  // We are now pointing to the wire group _after_ the required one
  // (unless we are at begin() or end() when we just return wireG=0)
  ptrdiff_t pd = it - theFirstWireOfEachWireGroup.begin();

  if (pd > 0) {
    size_t id = --pd;  //@@ Need to step back one. IS THIS SANE CODE?
    int wiresInGroup = theNumberOfWiresPerWireGroup[id];
    int firstWire = theFirstWireOfEachWireGroup[id];

    // Require wire not past end of group (may be in a dead region, or
    // bigger than total wires in chamber)
    if (wire < (firstWire + wiresInGroup))
      wireG = ++id;
  }
  return wireG;
}

float CSCGangedWireGrouping::middleWireOfGroup(int wireGroup) const {
  // Return exact wire number for group with odd number of wires
  // Return half-integer wire number for group with even number of wires
  // i.e. first + 0.5 * float(#) - 0.5
  // Return 0 if out of range

  float middleWire = 0.;
  if (wireGroup > 0 && wireGroup <= theNumberOfGroups) {
    middleWire = theFirstWireOfEachWireGroup[wireGroup - 1] + 0.5 * theNumberOfWiresPerWireGroup[wireGroup - 1] - 0.5;
  }
  return middleWire;
}
