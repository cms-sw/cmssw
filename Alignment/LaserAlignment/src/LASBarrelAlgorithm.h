
#ifndef __LASBARRELALGORITHM_H
#define __LASBARRELALGORITHM_H

#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>

#include <TMinuit.h>

#include "Alignment/LaserAlignment/src/LASBarrelAlignmentParameterSet.h"
#include "Alignment/LaserAlignment/src/LASCoordinateSet.h"
#include "Alignment/LaserAlignment/src/LASGlobalData.cc"
#include "Alignment/LaserAlignment/src/LASGlobalLoop.h"


///
/// implementation of the alignment tube algorithm
///
class LASBarrelAlgorithm {
  
 public:
  LASBarrelAlgorithm();
  LASBarrelAlignmentParameterSet CalculateParameters( LASGlobalData<LASCoordinateSet>&, LASGlobalData<LASCoordinateSet>& );
  void Dump( void );
  
 private:
  void ReadMisalignmentFromFile( const char*, LASGlobalData<LASCoordinateSet>&, LASGlobalData<LASCoordinateSet>& );
  TMinuit* minuit;

};

// minuit chisquare function
void fcn( int&, double*, double&, double*, int );

#endif
