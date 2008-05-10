
#ifndef __LASPROFILEJUDGE_H
#define __LASPROFILEJUDGE_H

#include <iostream>
#include <utility>

#include "Alignment/LaserAlignment/src/LASModuleProfile.h"


///
/// check if a LASModuleProfile is usable
/// for being stored and fitted
///
class LASProfileJudge {

 public:
  LASProfileJudge();
  bool IsSignalIn( const LASModuleProfile&, int );
  bool JudgeProfile( const LASModuleProfile&, int );
  void EnableZeroFilter( bool );

 private:
  double GetNegativity( int );
  bool IsPeaksInProfile( int );
  bool IsNegativePeaksInProfile( int );
  LASModuleProfile profile;
  std::pair<unsigned int, double> thePeak;
  bool isZeroFilter;

};

#endif
