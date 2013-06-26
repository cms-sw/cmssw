
#ifndef __LASBARRELALGORITHM_H
#define __LASBARRELALGORITHM_H

#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <TMinuit.h>

#include "Alignment/LaserAlignment/interface/LASBarrelAlignmentParameterSet.h"
#include "Alignment/LaserAlignment/interface/LASCoordinateSet.h"
#include "Alignment/LaserAlignment/interface/LASGlobalData.h"
#include "Alignment/LaserAlignment/interface/LASGlobalLoop.h"


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
  void ReadStartParametersFromFile( const char*, float[52] );
  TMinuit* minuit;

};

// minuit chisquare function
void fcn( int&, double*, double&, double*, int );

#endif
