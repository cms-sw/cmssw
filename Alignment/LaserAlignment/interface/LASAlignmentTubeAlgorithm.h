

#ifndef __LASALIGNMENTTUBEALGORITHM_H
#define __LASALIGNMENTTUBEALGORITHM_H

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include "Alignment/LaserAlignment/interface/LASBarrelAlignmentParameterSet.h"
#include "Alignment/LaserAlignment/interface/LASCoordinateSet.h"
#include "Alignment/LaserAlignment/interface/LASGlobalData.h"
#include "Alignment/LaserAlignment/interface/LASGlobalLoop.h"

///
/// implementation of the alignment tube analytical algorithm
///
class LASAlignmentTubeAlgorithm {
public:
  LASAlignmentTubeAlgorithm();
  LASBarrelAlignmentParameterSet CalculateParameters(LASGlobalData<LASCoordinateSet>&,
                                                     LASGlobalData<LASCoordinateSet>&);
  double GetTIBTOBAlignmentParameterCorrection(
      int, int, int, LASGlobalData<LASCoordinateSet>&, LASBarrelAlignmentParameterSet&);
  double GetTEC2TECAlignmentParameterCorrection(
      int, int, int, LASGlobalData<LASCoordinateSet>&, LASBarrelAlignmentParameterSet&);
  void ReadMisalignmentFromFile(const char*, LASGlobalData<LASCoordinateSet>&, LASGlobalData<LASCoordinateSet>&);

private:
};

#endif
