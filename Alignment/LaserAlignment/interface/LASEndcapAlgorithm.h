
#ifndef __LASENDCAPALGORITHM_H
#define __LASENDCAPALGORITHM_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include <TMinuit.h>

#include "Alignment/LaserAlignment/interface/LASEndcapAlignmentParameterSet.h"
#include "Alignment/LaserAlignment/interface/LASCoordinateSet.h"
#include "Alignment/LaserAlignment/interface/LASGlobalData.h"
#include "Alignment/LaserAlignment/interface/LASGlobalLoop.h"

///
/// calculate parameters for both endcaps from measurement
///
/// TODO: 
///   * calculate the parameter errors
///   * include the beam parameters
///   * calculate the global parameters
///
class LASEndcapAlgorithm {

 public:
  LASEndcapAlgorithm();
  LASEndcapAlignmentParameterSet CalculateParameters( LASGlobalData<LASCoordinateSet>&, LASGlobalData<LASCoordinateSet>& );
  double GetAlignmentParameterCorrection( int, int, int, int, LASGlobalData<LASCoordinateSet>&, LASEndcapAlignmentParameterSet& );

};



#endif
