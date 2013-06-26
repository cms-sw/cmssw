
#ifndef __LASPROFILEJUDGE_H
#define __LASPROFILEJUDGE_H

#include <iostream>
#include <utility>

#include "Alignment/LaserAlignment/interface/LASModuleProfile.h"


///
/// check if a LASModuleProfile is usable
/// for being stored and fitted
///
class LASProfileJudge {

 public:
  LASProfileJudge();
  bool IsSignalIn( const LASModuleProfile&, double );
  bool JudgeProfile( const LASModuleProfile&, double );
  void EnableZeroFilter( bool );
  void SetOverdriveThreshold( unsigned int );

 private:
  double GetNegativity( int );
  bool IsPeaksInProfile( int );
  bool IsNegativePeaksInProfile( int );
  bool IsOverdrive( int );

  LASModuleProfile profile;
  std::pair<unsigned int, double> thePeak;
  bool isZeroFilter;
  unsigned int overdriveThreshold;

};

#endif
