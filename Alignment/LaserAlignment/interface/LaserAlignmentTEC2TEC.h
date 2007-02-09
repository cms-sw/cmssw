/* 
 * Alignment of TEC-TIB-TOB-TEC
 */

#ifndef LaserAlignment_LaserAlignmentTEC2TEC_h
#define LaserAlignment_LaserAlignmentTEC2TEC_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/LaserAlignment/interface/LaserAlignmentAlgorithmTEC2TEC.h"

// Alignable Tracker needed to propagate the alignment corrections calculated 
// for the disks down to the lowest levels
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include <iostream>
#include <vector>

class LaserAlignmentTEC2TEC
{
 public:
  LaserAlignmentTEC2TEC();
  ~LaserAlignmentTEC2TEC();
  
  void alignment(edm::ParameterSet const & theConf, 
		 AlignableTracker * theAlignableTracker,
		 int theNumberOfIterations, int theAlignmentIteration,
		 std::vector<double>& theLaserPhi, 
		 std::vector<double>& theLaserPhiError);

 private:
  LaserAlignmentAlgorithmTEC2TEC * theLaserAlignmentTrackerTEC2TEC;
};
#endif
