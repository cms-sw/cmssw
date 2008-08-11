
#ifndef __LASENDCAPALGORITHM_H
#define __LASENDCAPALGORITHM_H

#include <iostream>
#include <vector>
#include <cmath>

#include <TMinuit.h>

#include "Alignment/LaserAlignment/src/LASEndcapAlignmentParameterSet.h"
#include "Alignment/LaserAlignment/src/LASCoordinateSet.h"
#include "Alignment/LaserAlignment/src/LASGlobalData.cc"
#include "Alignment/LaserAlignment/src/LASGlobalLoop.h"

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

};



#endif
