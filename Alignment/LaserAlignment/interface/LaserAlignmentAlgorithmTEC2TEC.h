#ifndef LaserAlignment_LaserAlignmentAlgorithmTEC2TEC_h
#define LaserAlignment_LaserAlignmentAlgorithmTEC2TEC_h

/** \class LaserAlignmentAlgorithmTEC2TEC
 *  class to align the tracker (TEC-TIB-TOB-TEC) with Millepede
 *
 *  $Date: Fri Mar 16 15:45:46 CET 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/interface/Millepede.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


// Alignable Tracker needed to propagate the alignment corrections calculated 
// for the disks down to the lowest levels
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include <iostream>
#include <vector>

class LaserAlignmentAlgorithmTEC2TEC 
{

 public:
  /// constructor
  LaserAlignmentAlgorithmTEC2TEC(edm::ParameterSet const& theConf, int theLaserIteration);

  /// destructor
  ~LaserAlignmentAlgorithmTEC2TEC();

  /// add a LaserBeam to Millepede and do a local fit
  void addLaserBeam(std::vector<double> theMeasurementsPosTEC, std::vector<double> theMeasurementsTOB,
		    std::vector<double> theMeasurementsTIB, std::vector<double> theMeasurementsNegTEC,
		    int LaserBeam, int LaserRing);
  /// do the global fit
  void doGlobalFit(AlignableTracker * theAlignableTracker);

  /// reset Millepede
  void resetMillepede(int UnitForIteration);

 private:
  /// initialize Millepede
  void initMillepede(int UnitForIteration);

 private:
  // arrays to hold the global and local parameters
  int theFirstFixedDiskTEC2TEC;
  int theSecondFixedDiskTEC2TEC;

  /* ATTENTION we have more than 27 parameters in this case! */
  float theGlobalParametersTEC2TEC[66];
  float theLocalParametersTEC2TEC[1];

};
#endif
