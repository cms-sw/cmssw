#ifndef LaserAlignment_LaserAlignmentTEC2TEC_h
#define LaserAlignment_LaserAlignmentTEC2TEC_h

/** \class LaserAlignmentTEC2TEC
 *  Alignment of TEC-TIB-TOB-TEC
 *
 *  $Date: Fri Mar 16 15:50:26 CET 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

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
