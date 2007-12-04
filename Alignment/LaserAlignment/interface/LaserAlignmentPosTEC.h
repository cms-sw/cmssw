#ifndef LaserAlignmentTracker_LaserAlignmentPosTEC_h
#define LaserAlignmentTracker_LaserAlignmentPosTEC_h

/** \class LaserAlignmentPosTEC
 *  Alignment of TEC+
 *
 *  $Date: 2007/03/18 19:00:19 $
 *  $Revision: 1.2 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/interface/LaserAlignmentAlgorithmPosTEC.h"

// Alignable Tracker needed to propagate the alignment corrections calculated 
// for the disks down to the lowest levels
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include <iostream>
#include <vector>

class LaserAlignmentPosTEC
{
 public:
	/// constructor
  LaserAlignmentPosTEC();
  /// destructor
  ~LaserAlignmentPosTEC();

	/// do the alignment
  void alignment(edm::ParameterSet const & theConf, 
		 AlignableTracker * theAlignableTracker,
		 int theNumberOfIterations, int theAlignmentIteration,
		 std::vector<double>& theLaserPhi, 
		 std::vector<double>& theLaserPhiError);

 private:
  LaserAlignmentAlgorithmPosTEC * theLaserAlignmentTrackerPosTEC;

};
#endif
