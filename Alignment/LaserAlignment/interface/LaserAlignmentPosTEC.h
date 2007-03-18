#ifndef LaserAlignmentTracker_LaserAlignmentPosTEC_h
#define LaserAlignmentTracker_LaserAlignmentPosTEC_h

/** \class LaserAlignmentPosTEC
 *  Alignment of TEC+
 *
 *  $Date: Fri Mar 16 15:47:23 CET 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

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
