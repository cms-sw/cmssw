/* 
 * class to align the tracker (TEC+) with Millepede
 */

#ifndef LaserAlignment_LaserAlignmentAlgorithmPosTEC_h
#define LaserAlignment_LaserAlignmentAlgorithmPosTEC_H

#include "Alignment/LaserAlignment/interface/Millepede.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Alignable Tracker needed to propagate the alignment corrections calculated 
// for the disks down to the lowest levels
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include <iostream>
#include <vector>

class LaserAlignmentAlgorithmPosTEC 
{

 public:
  // constructor
  LaserAlignmentAlgorithmPosTEC(edm::ParameterSet const& theConf, int theAlignmentIteration);

  // destructor
  ~LaserAlignmentAlgorithmPosTEC();

  // add a LaserBeam to Millepede and do a local fit
  void addLaserBeam(std::vector<double> theMeasurements, int LaserBeam, int LaserRing);
  // do the global fit
  void doGlobalFit(AlignableTracker * theAlignableTracker);

  // reset Millepede
  void resetMillepede(int UnitForIteration);

 private:
  // initialize Millepede
  void initMillepede(int UnitForIteration);

 private:
  // arrays to hold the global and local parameters
  int theFirstFixedDiskPosTEC;
  int theSecondFixedDiskPosTEC;
  float theGlobalParametersPosTEC[27]; 
  float theLocalParametersPosTEC[1];

};
#endif
