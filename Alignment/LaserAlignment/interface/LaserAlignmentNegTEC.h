#ifndef LaserAlignment_LaserAlignmentNegTEC_h
#define LaserAlignment_LaserAlignmentNegTEC_h

/** \class LaserAlignmentNegTEC
 *  Alignment of TEC-
 *
 *  $Date: Fri Mar 16 15:46:24 CET 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

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
	/// constructor
  LaserAlignmentNegTEC();
	/// destructor
  ~LaserAlignmentNegTEC();

	/// do the alignment
  void alignment(edm::ParameterSet const & theConf, 
		 AlignableTracker * theAlignableTracker, 
		 int theNumberOfIterations, int theAlignmentIteration, 
		 std::vector<double>& theLaserPhi, 
		 std::vector<double>& theLaserPhiError);

 private:
  LaserAlignmentAlgorithmNegTEC * theLaserAlignmentTrackerNegTEC;
};
#endif
