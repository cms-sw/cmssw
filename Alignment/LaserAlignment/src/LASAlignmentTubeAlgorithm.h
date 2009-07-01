

#ifndef __LASALIGNMENTTUBEALGORITHM_H
#define __LASALIGNMENTTUBEALGORITHM_H

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include "Alignment/LaserAlignment/src/LASBarrelAlignmentParameterSet.h"
#include "Alignment/LaserAlignment/src/LASCoordinateSet.h"
#include "Alignment/LaserAlignment/src/LASGlobalData.cc"
#include "Alignment/LaserAlignment/src/LASGlobalLoop.h"


///
/// implementation of the alignment tube analytical algorithm
///
class LASAlignmentTubeAlgorithm {
  
 public:
  LASAlignmentTubeAlgorithm();
  LASBarrelAlignmentParameterSet CalculateParameters( LASGlobalData<LASCoordinateSet>&, LASGlobalData<LASCoordinateSet>& );
  void ReadMisalignmentFromFile( const char*, LASGlobalData<LASCoordinateSet>&, LASGlobalData<LASCoordinateSet>& );

 private:

};

#endif
