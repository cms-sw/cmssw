
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
  bool JudgeProfile( const LASModuleProfile& );

 private:
  double GetNegativity( void );
  bool IsPeaksInProfile( void );
  bool IsNegativePeaksInProfile( void );
  LASModuleProfile profile;
  std::pair<unsigned int, double> thePeak;

};

#endif
