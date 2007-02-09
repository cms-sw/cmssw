/* 
 * Alignment of TEC+
 */

#ifndef LaserAlignmentTracker_LaserAlignmentPosTEC_h
#define LaserAlignmentTracker_LaserAlignmentPosTEC_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/LaserAlignment/interface/LaserAlignmentAlgorithmPosTEC.h"

// Alignable Tracker needed to propagate the alignment corrections calculated 
// for the disks down to the lowest levels
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include <iostream>
#include <vector>

class LaserAlignmentPosTEC
{
 public:
  LaserAlignmentPosTEC();
  ~LaserAlignmentPosTEC();

  void alignment(edm::ParameterSet const & theConf, 
		 AlignableTracker * theAlignableTracker,
		 int theNumberOfIterations, int theAlignmentIteration,
		 std::vector<double>& theLaserPhi, 
		 std::vector<double>& theLaserPhiError);

 private:
  LaserAlignmentAlgorithmPosTEC * theLaserAlignmentTrackerPosTEC;

};
#endif
