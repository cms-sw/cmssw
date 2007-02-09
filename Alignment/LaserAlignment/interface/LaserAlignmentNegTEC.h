/* 
 * Alignment of TEC-
 */

#ifndef LaserAlignment_LaserAlignmentNegTEC_h
#define LaserAlignment_LaserAlignmentNegTEC_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/LaserAlignment/interface/LaserAlignmentAlgorithmNegTEC.h"

// Alignable Tracker needed to propagate the alignment corrections calculated 
// for the disks down to the lowest levels
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include <iostream>
#include <vector>

class LaserAlignmentNegTEC
{
 public:
  LaserAlignmentNegTEC();
  ~LaserAlignmentNegTEC();

  void alignment(edm::ParameterSet const & theConf, 
		 AlignableTracker * theAlignableTracker, 
		 int theNumberOfIterations, int theAlignmentIteration, 
		 std::vector<double>& theLaserPhi, 
		 std::vector<double>& theLaserPhiError);

 private:
  LaserAlignmentAlgorithmNegTEC * theLaserAlignmentTrackerNegTEC;
};
#endif
