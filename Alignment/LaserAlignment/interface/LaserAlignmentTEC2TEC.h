#ifndef LaserAlignment_LaserAlignmentTEC2TEC_h
#define LaserAlignment_LaserAlignmentTEC2TEC_h

/** \class LaserAlignmentTEC2TEC
 *  Alignment of TEC-TIB-TOB-TEC
 *
 *  $Date: 2007/03/18 19:00:19 $
 *  $Revision: 1.2 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/interface/LaserAlignmentAlgorithmTEC2TEC.h"

// Alignable Tracker needed to propagate the alignment corrections calculated 
// for the disks down to the lowest levels
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include <iostream>
#include <vector>

class LaserAlignmentTEC2TEC
{
 public:
	/// constructor
  LaserAlignmentTEC2TEC();
	/// destructor
  ~LaserAlignmentTEC2TEC();

  /// do the alignment
  void alignment(edm::ParameterSet const & theConf, 
		 AlignableTracker * theAlignableTracker,
		 int theNumberOfIterations, int theAlignmentIteration,
		 std::vector<double>& theLaserPhi, 
		 std::vector<double>& theLaserPhiError);

 private:
  LaserAlignmentAlgorithmTEC2TEC * theLaserAlignmentTrackerTEC2TEC;
};
#endif
