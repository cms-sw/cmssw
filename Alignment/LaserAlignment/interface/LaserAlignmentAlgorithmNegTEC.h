#ifndef LaserAlignment_LaserAlignmentAlgorithmNegTEC_h
#define LaserAlignment_LaserAlignmentAlgorithmNegTEC_h

/** \class LaserAlignmentAlgorithmNegTEC
 *  class to align the tracker (TEC-) with Millepede
 *
 *  $Date: 2007/03/18 19:00:19 $
 *  $Revision: 1.2 $
 *  \author Maarten Thomas
 */

// Alignable Tracker needed to propagate the alignment corrections calculated 
// for the disks down to the lowest levels
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include <iostream>
#include <vector>

class LaserAlignmentAlgorithmNegTEC 
{

 public:
  /// constructor
  LaserAlignmentAlgorithmNegTEC(edm::ParameterSet const& theConf, int theLaserIteration);

  /// destructor
  ~LaserAlignmentAlgorithmNegTEC();

  /// add a LaserBeam to Millepede and do a local fit
  void addLaserBeam(std::vector<double> theMeasurements, int LaserBeam, int LaserRing);
  /// do the global fit
  void doGlobalFit(AlignableTracker * theAlignableTracker);

  /// reset Millepede
  void resetMillepede(int UnitForIteration);

 private:
  /// initialize Millepede
  void initMillepede(int UnitForIteration);

 private:
  // arrays to hold the global and local parameters
  int theFirstFixedDiskNegTEC;
  int theSecondFixedDiskNegTEC;
  float theGlobalParametersNegTEC[27];
  float theLocalParametersNegTEC[1];

};
#endif
